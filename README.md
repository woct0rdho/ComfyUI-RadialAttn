# ComfyUI-RadialAttn

This repo is [RadialAttention](https://github.com/mit-han-lab/radial-attention) ported to ComfyUI native workflows. If you're using kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) rather than native workflows, then you can use their `WanVideoSetRadialAttention` node rather than this repo, and you still need to install the pip packages below.

This supports Wan 2.1 and 2.2 14B, both T2V and I2V.

## Installation

Here I list all the Windows wheels. On Linux I guess you know what to do.

1. Install [triton-windows](https://github.com/woct0rdho/triton-windows)
2. Install [SageAttention](https://github.com/woct0rdho/SageAttention/releases)
3. Install [SpargeAttention](https://github.com/woct0rdho/SpargeAttn/releases)
4. Install [flashinfer-windows](https://github.com/SystemPanic/flashinfer-windows/releases)
    * Currently FlashInfer only supports PyTorch 2.6, but it's mostly a placeholder for the purpose of this repo. If you're using another version of PyTorch, you can run:
    ```pwsh
    pip install --no-deps .\flashinfer_python-0.2.8-cp39-abi3-win_amd64.whl
    pip install cuda-python einops ninja numpy pynvml requests
    ```
5. git clone this repo to your `ComfyUI/custom_nodes/`

## Usage

Just connect your model to the `PatchRadialAttn` node. There's an [example workflow](https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/main/example_workflows/radial_attn.json) for Wan 2.2 14B I2V + GGUF + LightX2V LoRA + RadialAttn + `torch.compile`.

It's believed that skipping RadialAttn on the first layer (`dense_block = 1`) and the first time step (`dense_timestep = 1`) improves the quality.

RadialAttn requires specific video sizes and lengths. The 'number of video tokens' must be divisible by 128. For Wan 14B, this number is computed by `width/16 * height/16 * (length+3)/4`. See [video_token_num](https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/14ed41e2ef754dfd0fb7d0ea4eea5ed2293edb55/nodes.py#L180) for details.

(A misunderstanding is that the width and the height must be divisible by 128, but that's actually not the case.)
