# TensorRT for Web-UI

This extension for Web-UI allows you to export the Unet of Stable diffusion to TensorRT and generate images faster!

## Installation
This extension can be installed just like any other extension for Automatic1111:
1. Start the `webui.bat`
2. Select the `Extensions` tab and click on `Install from URL`
3. Copy the link to this repository and paste it into `URL for extension's git repository`
4. Click `Install`

### Requirements 
##### TensorRT for Web-UI will `automatically` install required dependencies and initialize itself:
* `TensorRT` (9.0 Pre-release version will be installed)
* `Polygraphy` (for Onnx parsing and storing)
* `Onnx` (export of networks)

## How to use
The webui will show a TensorRT Tab after installation. You have two options in this tab:
1. Convert to ONNX
2. Convert ONNX to TensorRT (Preset config)

### Convert to ONNX
This tab allows you to export the `current selected` Stable Diffusion checkpoint to Onnx. You only need to `export it once` as long as you dont change anything for the checkpoint. LoRAs that are inside will also be exported to Onnx, so remember to update your ONNX file when modifiying the checkpoint. 


### Convert ONNX to TensorRT (Preset config)
Here you can select an ONNX file (located in `models/Unet-onnx`) and convert it to TensorRT with selected presets. The presets are the labeled with `Width x Height x Batch_size`. You can select multiple presets and export them to TensorRT. Warning: This process takes a long time and is memory intensive, you can see some debug print in the terminal while converting. You will find TensorRT engines in `models/Unet-trt` once the conversion finished.

### Using TensorRT engines

In settings in the Webui, go to the `Stable Diffusion` tab and select your TensorRT engine under `SD Unet`