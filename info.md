# TensorRT

You can use this extension to export optimized TensorRT models for best performance on NVIDIA RTX GPUs. This process has to be done once for every model
atchitecture you want to use.

**Please Note** that this process can take between 3-15 minutes. Ideally the GPU should not be stressed during that process as it might
result in suboptimal engine generation.

## How To

1. Select the model you want to convert from the `Stable Diffusion checkpoint` dopdown in the main UI
2. Select a model preset from the `Stable Diffusion Version` dropdown in the extension.
3. **Optional:** Use advanced settings to fine tune for your needs. More on that can be found in the Advanced Settings section.
4. Export the model.
5. Once the model is exported you need to enable it:
   1. Go to Settings -> Stable Diffusion
   2. In the `SD U-Net` dropdown select the model you want to use.
6. Happy prompting.

## Advanced Settings

**Intro** TensorRT leverages an exhaustive search to find the best possible execution plan. To enable this, TensorRT need to know what input shapes it should optimize for as well as bound on the minimal and maximal input shapes.

All of these settings can be tewaked in the advanced seetings.

**Note:** Increasing the maximum dimensions will increase the compile time as well as the VRAM consumption of the final model. For best performance it is recommended to export an engine with the `Use Static Shapes` option enabled and setting the optimal shapes you intend to use.

## Tested Model

- SD 1.5
- SD 2.1 Base
- SD 2.1 Inpaint
