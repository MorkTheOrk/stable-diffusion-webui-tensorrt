# TensorRT Extension for Stable Diffusion 

This extension enables the best performance on NVIDIA RTX GPUs for Stable Diffusion with TensorRT. 

You need to install the extension and generate optimized engines before using the extension. Please follow the instructions below to set everything up. 

Supports Stable Diffusion 1.5, 2.1, and SDXL.

## Installation

Example instructions for Automatic1111:

1. Start the webui.bat
2. Select the Extensions tab and click on Install from URL
3. Copy the link to this repository and paste it into URL for extension's git repository
4. Click Install

## How to use

1. Click on the “Generate Default Engines” button. This step takes 2-10 minutes depending on your GPU. You can generate engines for other combinations. 
2. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
3. Back in the main UI, select the TRT model from the sd_unet dropdown menu at the top of the page. 
4. You can now start generating images accelerated by TRT. If you need to create more Engines, go to the TensorRT tab. 

Happy prompting!

## More Information

TensorRT uses optimized engines for specific resolutions and batch sizes. You can generate as many optimized engines as desired. Types:

- The “Generate Default Engines” selection adds support for resolutions between 512x512 and 768x768 for Stable Diffusion 1.5 and 768x768 to 1024x1024 for SDXL with batch sizes 1 to 4.
- Static engines support a single specific output resolution and batch size. 
- Dynamic engines support a range of resolutions and batch sizes, at a small cost in performance. Wider ranges will use more VRAM. 

Each preset can be adjusted with the “Advanced Settings” option.
