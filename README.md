# ComfyUI-RadialAttn

This repo is [RadialAttention](https://github.com/mit-han-lab/radial-attention) ported to ComfyUI native workflows. If you're using kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) rather than native workflows, then you can use their `WanVideoSetRadialAttention` node rather than this repo, and you still need to install the SpargeAttention wheel below.

This supports Wan 2.1 14B, Wan 2.2 14B, Wan 2.2 5B, and many of their variants, both T2V and I2V.

This does not give speedup if you only generate a single frame image.

## Installation

1. Install [SpargeAttention](https://github.com/woct0rdho/SpargeAttn)
2. git clone ComfyUI-RadialAttn to your `ComfyUI/custom_nodes/`

It's also recommended to install [SageAttention](https://github.com/woct0rdho/SageAttention), and add `--use-sage-attention` when starting ComfyUI. When RadialAttention is not applicable, SageAttention will be used.

## Usage

Just connect your model to the `PatchRadialAttn` node. There's an [example workflow](https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/main/example_workflows/radial_attn.json) for Wan 2.2 14B I2V + LightX2V LoRA + RadialAttention.

It's believed that disabling RadialAttention on the first layer (`dense_block = 1`), the first time step (`dense_timestep = 1`), and the last time step (`last_dense_timestep = 1`) improves the quality.
