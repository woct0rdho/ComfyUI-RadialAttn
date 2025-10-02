import functools
from unittest.mock import patch

import torch
from comfy.ldm.modules.attention import wrap_attn

from .attn_mask import MaskMap, RadialAttention
from .patches import _original_functions, patched_forward


@functools.cache
def get_radial_attn_func(video_token_num, num_frame, block_size, decay_factor):
    mask_map = MaskMap(video_token_num, num_frame)

    @torch.compiler.disable()
    @wrap_attn
    def radial_attn_func(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        # attn_precision is unused
        assert mask is None
        assert skip_reshape is False
        assert skip_output_reshape is False

        if q.shape != k.shape:
            # This is cross attn. Fallback to the original attn.
            orig_attention = _original_functions.get("orig_attention")
            return orig_attention(q, k, v, heads)

        # (batch_size, seq_len, num_heads * head_dim) -> (batch_size * seq_len, num_heads, head_dim)
        b, _, head_dim = q.shape
        head_dim //= heads
        q, k, v = map(lambda t: t.view(-1, heads, head_dim), (q, k, v))

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

        # (batch_size * seq_len, num_heads, head_dim) -> (batch_size, seq_len, num_heads * head_dim)
        # Cannot use view because out may not be contiguous
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

        type_name = type(diffusion_model).__name__
        if type_name in patched_forward:
            context = patch.multiple(diffusion_model, forward_orig=patched_forward[type_name].__get__(diffusion_model, diffusion_model.__class__))
        else:
            raise TypeError(f"Unsupported model: {type_name}")

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
            patch_size = ra_options["patch_size"]
            num_frame = (input.shape[2] - 1) // patch_size[0] + 1
            frame_size = (input.shape[3] // patch_size[1]) * (input.shape[4] // patch_size[2])
            video_token_num = frame_size * num_frame

            ra_options["radial_attn_func"] = get_radial_attn_func(video_token_num, num_frame, ra_options["block_size"], ra_options["decay_factor"])

            if ra_options["dense_timestep"] <= current_step_index < len(sigmas) - 1 - ra_options["last_dense_timestep"]:
                with context:
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
