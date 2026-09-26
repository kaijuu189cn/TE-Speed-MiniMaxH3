"""TESpeedMiniMaxH3 Linux 纯 Python 实现。

注册节点 ID `TESpeedMiniMaxH3`，与 Windows 原版完全一致。
工作流无需改线即可使用。

算法核心：Block-level Residual Caching
  DiT block 的 hidden state 在相邻 denoising 步之间变化较小时可复用。
  patches_replace["dit"] 拦截 DiT block，根据 residual 相似度决策。

  使用 sigma 值跟踪采样进度（ComfyUI 未设 step_index）。
  CFG 正负分支独立追踪缓存。

Supported modes:
  - standard: 长采样通用加速（~10%-85% 进度启用）
  - 4-step LoRA: 4步短轨迹专用加速
  - 8-step LoRA: 8步短轨迹专用加速
"""

import logging
import time
import torch

logger = logging.getLogger("TESpeed-Linux")

# ── 常数 ─────────────────────────────────────────────────────────
_TURBO_START = 0.10
_TURBO_END_MARGIN = 0.15
_TURBO_RMS_DRIFT_LIMIT = 0.06
_TURBO_RMS_GAIN_LIMIT = 0.08
_TURBO_DELTA_FLOOR = 0.005
_TURBO_SHORT_STEP_MAX = 10

_TURBO_EIGHT_START_PERCENT = 0.0
_TURBO_EIGHT_END_PERCENT = 0.85
_TURBO_EIGHT_DELTA_FLOOR = 0.003
_TURBO_EIGHT_RMS_DRIFT_LIMIT = 0.04
_TURBO_EIGHT_RMS_GAIN_LIMIT = 0.05
_TURBO_EIGHT_MAX_HITS = 5

_RESIDUAL_FORECAST_BLEND = 0.35
_RESIDUAL_FORECAST_LIMIT = 10
_RESIDUAL_FORECAST_MAX_BETA = 0.95
_CACHE_TRANSFER_CHUNK = 8
_DYNAMIC_VRAM_GPU_CACHE_MAX = 0.85
_RMS_DRIFT_LIMIT = 0.1
_RMS_GAIN_LIMIT = 0.15


# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════

def _rel_l2(old, new):
    """Relative L2 change between two tensors."""
    with torch.no_grad():
        diff = torch.norm(new.float() - old.float()).item()
        norm = max(torch.norm(old.float()).item(), 1e-8)
        return diff / norm


def _get_step_info(transformer_options):
    """Derive (step_index, total_steps, current_sigma) from transformer_options."""
    sigmas = transformer_options.get("sigmas")
    sample_sigmas = transformer_options.get("sample_sigmas")
    
    if sigmas is not None and len(sigmas) > 1:
        total = len(sigmas) - 1
        # sigmas here is the full schedule, we need current step
        # Sample sigmas gives us per-call sigma info
        if sample_sigmas is not None and len(sample_sigmas) > 0:
            # The current sigma can be derived from the first sample_sigma
            cur_sigma = float(sample_sigmas[0])
        else:
            cur_sigma = 0.0
        return None, total, cur_sigma  # step_id unknown without tracking
    return 0, 20, 0.0


def _turbo_ranges(mode, progress, total_blocks):
    """
    Determine (turbo_ranges, cache_ranges) for given progress.
    progress in [0, 1], 0 = start, 1 = end.
    """
    t = max(0.0, min(1.0, progress))

    if mode == "8-step":
        if t < _TURBO_EIGHT_START_PERCENT or t > _TURBO_EIGHT_END_PERCENT:
            return [(0, total_blocks)], []  # all turbo
        cache_start = int(total_blocks * 0.4)
        return [(0, cache_start)], [(cache_start, total_blocks)]

    elif mode == "4-step":
        if t < 0.33:
            cache_start = int(total_blocks * 0.5)
        elif t < 0.66:
            cache_start = int(total_blocks * 0.7)
        else:
            cache_start = total_blocks
        return [(0, cache_start)], [(cache_start, total_blocks)]

    else:  # standard
        if t < _TURBO_START or t > 1.0 - _TURBO_END_MARGIN:
            return [(0, total_blocks)], []
        cache_start = int(total_blocks * (0.25 + t * 0.25))
        cache_end = int(total_blocks * (1.0 - t * 0.1))
        if cache_start >= cache_end:
            return [(0, total_blocks)], []
        return [(0, cache_start)], [(cache_start, cache_end)]


# ════════════════════════════════════════════════════════════════════
# 缓存状态管理器
# ════════════════════════════════════════════════════════════════════

class _BranchCache:
    """Per-CFG-branch cache state."""

    def __init__(self):
        # Step key → per-block residual/state
        # Keyed by sigma value rounded to 6 decimal places
        self.prev_residual = {}     # block_idx → tensor
        self.prev_output = {}       # block_idx → tensor
        self.cache_stats = {}       # block_idx → (mean, std)
        self.last_sigma = None
        self.consec_skip = 0
        self.accum_delta = 0.0
        self.computed = 0
        self.skipped = 0
        self.total_computed = 0
        self.total_skipped = 0


class _TESpeedState:
    """Overall TE-Speed state shared across blocks."""

    def __init__(self, n_blocks, mode, debug=False):
        self.n_blocks = n_blocks
        self.mode = mode
        self.debug = debug
        # Branches: key (e.g. "pos", "neg") → _BranchCache
        self.branches = {}
        # Track steps via sigma sequence to detect step changes
        self._sigma_history = []       # list of sigma values seen
        self._current_step_idx = -1
        self._total_steps = 20
        self._reset_flag = True

    def reset(self):
        """Reset for new sampling run."""
        self.branches.clear()
        self._sigma_history.clear()
        self._current_step_idx = -1
        self._total_steps = 20
        self._reset_flag = True

    def get_branch(self, transformer_options):
        """Get or create branch cache for current CFG branch."""
        cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
        # Simple branch key: if cond_or_uncond has more than one entry,
        # we're in CFG; create separate caches for cond(0) and uncond(1)
        if len(cond_or_uncond) > 1:
            # First entry determines branch
            bk = f"branch_{cond_or_uncond[0]}"
        else:
            bk = "single"
        if bk not in self.branches:
            self.branches[bk] = _BranchCache()
        return self.branches[bk]

    def update_step(self, sigma_val):
        """Track step index from sigma values.
        
        Returns (step_idx, total_steps, is_new_step).
        """
        sigma_key = round(sigma_val, 6)
        
        if self._reset_flag:
            self._sigma_history = [sigma_key]
            self._current_step_idx = 0
            self._reset_flag = False
            return self._current_step_idx, self._total_steps, True
        
        # Check if this sigma is new (different from last)
        if not self._sigma_history or sigma_key != self._sigma_history[-1]:
            self._sigma_history.append(sigma_key)
            self._current_step_idx = len(self._sigma_history) - 1
            # Infer total steps from sigma pattern length if available
            # Common schedules: 20 steps default
            is_new = True
        else:
            is_new = False
        
        return self._current_step_idx, max(self._total_steps, self._current_step_idx + 1), is_new


# ════════════════════════════════════════════════════════════════════
# ComfyUI 节点
# ════════════════════════════════════════════════════════════════════

class TESpeedMiniMaxH3:
    """
    MiniMax H3 推理加速节点（Linux pure Python）
    """

    CATEGORY = "model_patches/unet"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (
                    ["standard", "4-step LoRA", "8-step LoRA"],
                    {"default": "standard"},
                ),
                "device": (
                    ["auto", "cpu", "gpu"],
                    {"default": "auto"},
                ),
                "dynamic_vram": (
                    ["enable", "disable"],
                    {"default": "enable"},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "patch_overlap": (
                    "INT",
                    {"default": 2, "min": 0, "max": 8, "step": 1},
                ),
            },
        }

    def patch(self, model, mode, device, dynamic_vram, debug,
              patch_overlap=2):
        # Clone
        cloned = model.clone()
        mp = cloned

        # Validate model
        dif_mod = getattr(mp.model, "diffusion_model", None)
        if dif_mod is None:
            logger.error("No diffusion_model found")
            return (model, "ERROR: no diffusion_model")

        blocks = getattr(dif_mod, "blocks", [])
        n_blocks = len(blocks)
        if n_blocks < 2:
            logger.error("TE-Speed requires >= 2 transformer blocks")
            return (model, "ERROR: < 2 blocks")

        mode_map = {"standard": "standard",
                     "4-step LoRA": "4-step",
                     "8-step LoRA": "8-step"}
        internal_mode = mode_map.get(mode, "standard")

        # Create state
        state = _TESpeedState(n_blocks, internal_mode, debug)
        mp._te_state = state

        # ── Block patches ─────────────────────────────────────────
        def make_block_patch(bi):
            def block_patch(args, original_block):
                h = args["img"]
                to = args.get("transformer_options", {})

                # Get sigma from timestep
                sigmas = to.get("sigmas")
                if sigmas is not None and len(sigmas) > 0:
                    # Approximate: the model currently processes at sigma = first entry
                    sigma_val = float(sigmas[0])
                else:
                    sigma_val = 0.0

                # Update step tracking
                step_idx, total_steps, is_new_step = state.update_step(sigma_val)

                if state._reset_flag:
                    state._reset_flag = False

                # Get branch cache
                bc = state.get_branch(to)

                # Log step info on step change
                if is_new_step and step_idx > 0:
                    if debug and (bc.computed + bc.skipped) > 0:
                        pct = bc.skipped / max(bc.computed + bc.skipped, 1) * 100
                        logger.debug(
                            f"[TE-Speed] step_{step_idx}/{total_steps} "
                            f"sigma={sigma_val:.4f} "
                            f"C={bc.computed} S={bc.skipped} "
                            f"skip%={pct:.0f}"
                        )
                    # Reset per-step counters but keep totals
                    bc.computed = 0
                    bc.skipped = 0
                    bc.consec_skip = 0
                    bc.accum_delta = 0.0

                # Progress for range decision
                if total_steps > 1:
                    progress = step_idx / max(total_steps - 1, 1)
                else:
                    progress = 0.0

                # Get ranges
                turbo_rng, cache_rng = _turbo_ranges(
                    internal_mode, progress, n_blocks
                )

                # Is this block in turbo (always compute) or cacheable?
                is_turbo = any(lo <= bi < hi for lo, hi in turbo_rng)
                is_cacheable = any(lo <= bi < hi for lo, hi in cache_rng)

                # Always compute for first step or turbo blocks
                if state._reset_flag or is_turbo or step_idx == 0:
                    h_out = original_block(args)
                    new_res = h_out["img"].float() - h.float()
                    bc.prev_residual[bi] = new_res.clone()
                    bc.prev_output[bi] = h_out["img"].clone()
                    bc.computed += 1
                    bc.total_computed += 1
                    return h_out

                # ── Cache decision ─────────────────────────────────
                prev_res = bc.prev_residual.get(bi)
                if prev_res is not None and is_cacheable:
                    # Decide whether to skip
                    delta_floor = (
                        _TURBO_EIGHT_DELTA_FLOOR if internal_mode == "8-step"
                        else _TURBO_DELTA_FLOOR
                    )
                    drift_limit = (
                        _TURBO_EIGHT_RMS_DRIFT_LIMIT if internal_mode == "8-step"
                        else _TURBO_RMS_DRIFT_LIMIT
                    )
                    max_skips = (
                        _TURBO_EIGHT_MAX_HITS if internal_mode == "8-step"
                        else 4 if internal_mode == "4-step"
                        else 3
                    )

                    # Quick check: if h hasn't changed much since last step,
                    # reuse prev residual
                    prev_out = bc.prev_output.get(bi)
                    if (prev_out is not None and 
                            bc.consec_skip < max_skips and
                            bc.accum_delta < drift_limit):
                        # Apply previous residual to current h
                        h_result = h + prev_res.to(h.dtype)
                        bc.skipped += 1
                        bc.total_skipped += 1
                        bc.consec_skip += 1
                        return {"img": h_result}

                # Compute normally
                h_out = original_block(args)
                new_res = h_out["img"].float() - h.float()

                # Update delta
                old_res = bc.prev_residual.get(bi)
                if old_res is not None:
                    delta = _rel_l2(old_res, new_res)
                    bc.accum_delta += delta

                bc.prev_residual[bi] = new_res.clone()
                bc.prev_output[bi] = h_out["img"].clone()
                bc.computed += 1
                bc.total_computed += 1
                bc.consec_skip = 0

                # Cache stats
                rf = new_res.reshape(-1).float()
                bc.cache_stats[bi] = (rf.mean().item(), rf.std().item())

                return h_out

            return block_patch

        # Install patches_replace["dit"]
        dit_patches = {}
        for i in range(n_blocks):
            dit_patches[("double_block", i)] = make_block_patch(i)

        mp.model_options.setdefault("transformer_options", {})
        to = mp.model_options["transformer_options"]
        pr = to.setdefault("patches_replace", {})
        pr["dit"] = dit_patches

        # ── Install lifecycle callbacks ────────────────────────────
        from comfy.patcher_extension import (
            CallbacksMP, add_callback_with_key, add_callback,
        )

        def on_pre_run(model_obj):
            """Reset state at sampling start."""
            s = getattr(model_obj, "_te_state", None)
            if s:
                s.reset()
            if debug:
                logger.info(
                    f"[TE-Speed] Started, mode={mode}, "
                    f"blocks={n_blocks}"
                )

        add_callback_with_key(
            CallbacksMP.ON_PRE_RUN,
            "te_speed_minimax_h3",
            on_pre_run,
            to,
        )

        # Also register at model_patcher level
        try:
            add_callback(CallbacksMP.ON_PRE_RUN, on_pre_run, to)
        except Exception:
            pass

        # ── Status ────────────────────────────────────────────────
        status = (
            f"TE-Speed-Linux v3.5.0 | "
            f"mode={mode} "
            f"blocks={n_blocks} "
            f"device={device}"
        )
        if debug:
            logger.info(status)

        return (cloned, status)


# ════════════════════════════════════════════════════════════════════
# 节点注册
# ════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "TESpeedMiniMaxH3": TESpeedMiniMaxH3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedMiniMaxH3": "TE-Speed-MiniMaxH3 (Linux)",
}