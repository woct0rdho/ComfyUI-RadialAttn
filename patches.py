import comfy
import torch
from comfy.ldm.wan.model import sinusoidal_embedding_1d
from tqdm import tqdm

_initialized = False
_original_functions = {}
if not _initialized:
    _original_functions["orig_attention"] = comfy.ldm.modules.attention.optimized_attention
    _initialized = True


patched_forward = {}


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

    full_ref = None
    if self.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

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
                out["img"] = block(
                    args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap}
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

    tqdm.write("Unapply radial attn")
    comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

    # head
    x = self.head(x, e)

    if full_ref is not None:
        x = x[:, full_ref.shape[1] :]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


patched_forward["WanModel"] = _WanModel_forward_orig


def _HumoWanModel_forward_orig(
    self,
    x,
    t,
    context,
    freqs=None,
    audio_embed=None,
    reference_latent=None,
    transformer_options={},
    **kwargs,
):
    ra_options = transformer_options["radial_attn"]

    bs, _, time, height, width = x.shape

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    if reference_latent is not None:
        ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
        ref = ref.flatten(2).transpose(1, 2)
        freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=time, device=x.device, dtype=x.dtype)
        x = torch.cat([x, ref], dim=1)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        del ref, freqs_ref

    # context
    context = self.text_embedding(context)
    context_img_len = None

    if audio_embed is not None:
        if reference_latent is not None:
            zero_audio_pad = torch.zeros(audio_embed.shape[0], reference_latent.shape[-3], *audio_embed.shape[2:], device=audio_embed.device, dtype=audio_embed.dtype)
            audio_embed = torch.cat([audio_embed, zero_audio_pad], dim=1)
        audio = self.audio_proj(audio_embed).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    else:
        audio = None

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
                out["img"] = block(
                    args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, audio=audio, transformer_options=args["transformer_options"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap}
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, audio=audio, transformer_options=transformer_options)

    tqdm.write("Unapply radial attn")
    comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


patched_forward["HumoWanModel"] = _HumoWanModel_forward_orig


def _AnimateWanModel_forward_orig(
    self,
    x,
    t,
    context,
    clip_fea=None,
    pose_latents=None,
    face_pixel_values=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    ra_options = transformer_options["radial_attn"]

    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    full_ref = None
    if self.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

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
                out["img"] = block(
                    args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"]
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap}
            )
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        if i % 5 == 0 and motion_vec is not None:
            # Disable radial_attn on face_adapter
            _optimized_attention = comfy.ldm.wan.model.optimized_attention
            _optimized_attention = _original_functions.get("orig_attention")

            x = x + self.face_adapter.fuser_blocks[i // 5](x, motion_vec)

            comfy.ldm.wan.model.optimized_attention = _optimized_attention

    tqdm.write("Unapply radial attn")
    comfy.ldm.wan.model.optimized_attention = _original_functions.get("orig_attention")

    # head
    x = self.head(x, e)

    if full_ref is not None:
        x = x[:, full_ref.shape[1] :]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


patched_forward["AnimateWanModel"] = _AnimateWanModel_forward_orig
