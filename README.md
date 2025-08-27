# ComfyUI-RadialAttn

This repo is [RadialAttention](https://github.com/mit-han-lab/radial-attention) ported to ComfyUI native workflows. If you're using kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) rather than native workflows, then you can use their `WanVideoSetRadialAttention` node rather than this repo, and you still need to install the pip packages below.

This supports Wan 2.1 14B, Wan 2.2 14B, Wan 2.2 5B, both T2V and I2V. This does not give speedup if you only generate a single frame image.

## Installation

1. Install [SpargeAttention](https://github.com/woct0rdho/SpargeAttn/releases)
2. git clone this repo to your `ComfyUI/custom_nodes/`

It's also recommended to install [SageAttention](https://github.com/woct0rdho/SageAttention/releases), and add `--use-sage-attention` when starting ComfyUI. When RadialAttention is not applicable, SageAttention will be used.

## Usage

Just connect your model to the `PatchRadialAttn` node. There's an [example workflow](https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/main/example_workflows/radial_attn.json) for Wan 2.2 14B I2V + GGUF + LightX2V LoRA + RadialAttention + `torch.compile`.

It's believed that skipping RadialAttention on the first layer (`dense_block = 1`) and the first time step (`dense_timestep = 1`) improves the quality.

RadialAttention requires specific video sizes and lengths:
* The 'number of video tokens' must be divisible by 128, see [video_token_num](https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/14ed41e2ef754dfd0fb7d0ea4eea5ed2293edb55/nodes.py#L180) for details
* For Wan 2.1 and 2.2 14B, this number is computed by `width/16 * height/16 * (length+3)/4`
* For Wan 2.2 5B, this number is computed by `width/32 * height/32 * (length+3)/4`

(A misunderstanding is that the width and the height must be divisible by 128, but that's actually not the case.)

Don't blindly use `torch.compile`. To start with, you can disable the `TorchCompileModel` node and run the workflow. Only when you're sure that the workflow runs but it's not fast enough, then you can try to enable `TorchCompileModel`. There are reports that `torch.compile` is slower in PyTorch 2.8 .
