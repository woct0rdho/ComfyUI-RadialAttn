import functools
from unittest.mock import patch

import torch
from torch.nn import functional as F

from comfy.ldm.modules.attention import optimized_attention, wrap_attn

from .attn_mask import MaskMap, RadialAttention


@functools.cache
def get_radial_attn_func(video_token_num, num_frame, block_size, decay_factor):
    mask_map = MaskMap(video_token_num, num_frame)

    @torch.compiler.disable()
    @wrap_attn
    def radial_attn_func(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if q.shape != k.shape:
            # This is cross attn. Fallback to the original attn.
            return optimized_attention(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        # attn_precision is unused
        assert mask is None

        if skip_reshape:
            # (batch_size, num_heads, seq_len, head_dim) -> (batch_size * seq_len, num_heads, head_dim)
            b, _, orig_seq_len, head_dim = q.shape
            q, k, v = map(lambda t: t.permute(0, 2, 1, 3).reshape(-1, heads, head_dim), (q, k, v))
        else:
            # (batch_size, seq_len, num_heads * head_dim) -> (batch_size * seq_len, num_heads, head_dim)
            b, orig_seq_len, head_dim = q.shape
            head_dim //= heads
            q, k, v = map(lambda t: t.view(-1, heads, head_dim), (q, k, v))

        padded_len = b * video_token_num
        if q.shape[0] != padded_len:
            q, k, v = map(lambda t: F.pad(t, (0, 0, 0, 0, 0, padded_len - t.shape[0])), (q, k, v))

        out = RadialAttention(
            q,
            k,
            v,
            mask_map=mask_map,
            sparsity_type="radial",
            block_size=block_size,
            decay_factor=decay_factor,
            model_type="wan",
            pre_defined_mask=None,
            use_sage_attention=True,
        )

        out = out[: b * orig_seq_len, :, :]

        if skip_output_reshape:
            # (batch_size * seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
            out = out.reshape(b, orig_seq_len, heads, head_dim).permute(0, 2, 1, 3)
        else:
            # (batch_size * seq_len, num_heads, head_dim) -> (batch_size, seq_len, num_heads * head_dim)
            out = out.reshape(b, -1, heads * head_dim)
        return out

    return radial_attn_func


class PatchRadialAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "dense_block": ("INT", {"default": 1, "min": 0, "max": 40, "step": 1, "tooltip": "Number of the first few blocks to disable radial attn."}),
                "dense_timestep": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "Number of the first few time steps to disable radial attn."}),
                "last_dense_timestep": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "Number of the last few time steps to disable radial attn."}),
                "block_size": ([64, 128], {"default": 128}),
                "decay_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Lower is faster, higher is more accurate."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_radial_attn"
    CATEGORY = "RadialAttn"

    def patch_radial_attn(self, model, dense_block, dense_timestep, last_dense_timestep, block_size, decay_factor):
        model = model.clone()

        diffusion_model = model.get_model_object("diffusion_model")

        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        if "radial_attn" not in model.model_options["transformer_options"]:
            model.model_options["transformer_options"]["radial_attn"] = {}
        ra_options = model.model_options["transformer_options"]["radial_attn"]

        ra_options["patch_size"] = diffusion_model.patch_size
        ra_options["dense_block"] = dense_block
        ra_options["dense_timestep"] = dense_timestep
        ra_options["last_dense_timestep"] = last_dense_timestep
        ra_options["block_size"] = block_size
        ra_options["decay_factor"] = decay_factor

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            sigmas = c["transformer_options"]["sample_sigmas"]

            matched_step_index = (sigmas == timestep).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
                else:
                    current_step_index = 0

            ra_options = c["transformer_options"]["radial_attn"]

            if ra_options["dense_timestep"] <= current_step_index < len(sigmas) - 1 - ra_options["last_dense_timestep"]:
                patch_size = ra_options["patch_size"]
                num_frame = (input.shape[2] - 1) // patch_size[0] + 1
                frame_size = (input.shape[3] // patch_size[1]) * (input.shape[4] // patch_size[2])
                video_token_num = frame_size * num_frame

                padded_video_token_num = video_token_num
                if video_token_num % block_size != 0:
                    padded_video_token_num = (video_token_num // block_size + 1) * block_size

                dense_block = ra_options["dense_block"]
                radial_attn_func = get_radial_attn_func(padded_video_token_num, num_frame, ra_options["block_size"], ra_options["decay_factor"])

                def maybe_radial_attn(*args, **kwargs):
                    transformer_options = kwargs.get("transformer_options", {})
                    block_index = transformer_options.get("block_index", -1)
                    if block_index >= dense_block:
                        return radial_attn_func(*args, **kwargs)
                    else:
                        return optimized_attention(*args, **kwargs)

                with patch("comfy.ldm.wan.model.optimized_attention", new=maybe_radial_attn):
                    return model_function(input, timestep, **c)
            else:
                # Do not apply radial attn
                return model_function(input, timestep, **c)

        model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "PatchRadialAttn": PatchRadialAttn,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchRadialAttn": "PatchRadialAttn",
}
