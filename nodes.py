import functools
from unittest.mock import patch

import comfy
import torch
from comfy.ldm.modules.attention import wrap_attn
from comfy.ldm.wan.model import WanModel, sinusoidal_embedding_1d
from tqdm import tqdm

from .attn_mask import MaskMap, RadialAttention

_initialized = False
_original_functions = {}
if not _initialized:
    _original_functions["orig_attention"] = comfy.ldm.modules.attention.optimized_attention
    _initialized = True


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


def _WanModel_forward_orig(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    ra_options = transformer_options["radial_attn"]

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    # Unapply before the 0-th layer in case it's not unapplied because of interrupt previously
    tqdm.write("Unapply radial attn")
    comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

    for i, block in enumerate(self.blocks):
        if i == ra_options["dense_block"]:
            tqdm.write(f"Apply radial attn from layer {i}")
            comfy.ldm.wan.model.optimized_attention = ra_options["radial_attn_func"]

        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                return out

            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

    tqdm.write("Unapply radial attn")
    comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


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
        assert type(diffusion_model) is WanModel

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

        context = patch.multiple(diffusion_model, forward_orig=_WanModel_forward_orig.__get__(diffusion_model, diffusion_model.__class__))

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
