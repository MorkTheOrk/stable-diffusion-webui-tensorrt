import html
import os

import launch
from modules import script_callbacks, paths_internal, shared
import gradio as gr

import export_onnx
import export_trt
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow


ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
    os.mkdir(ONNX_MODEL_DIR)
    
ONNX_MODELS = [ os.path.join(ONNX_MODEL_DIR, f"{m}") for m in os.listdir(ONNX_MODEL_DIR)]
    
                
def load_onnx_list(): 
    ONNX_MODELS = [ os.path.join(ONNX_MODEL_DIR, f"{m}") for m in os.listdir(ONNX_MODEL_DIR)]
    return gr.Dropdown.update(choices=ONNX_MODELS)

def export_unet_to_onnx(filename, opset):
    if not filename:
        modelname = shared.sd_model.sd_checkpoint_info.model_name + ".onnx"
        filename = os.path.join(paths_internal.models_path, "Unet-onnx", modelname)


    export_onnx.export_current_unet_to_onnx(filename, opset)

    return f'Saved as {filename}', ''


def get_trt_filename(filename, onnx_filename):
    if filename:
        return filename

    return os.path.join(paths_internal.models_path, "Unet-trt", "")



def convert_onnx_to_trt(filename, onnx_filename, *args):
    assert not cmd_opts.disable_extension_access, "won't run the command to create TensorRT file because extension access is dsabled (use --enable-insecure-extension-access)"
    filename = get_trt_filename(filename, onnx_filename)
    export_trt.generate_trt_engine(filename, onnx_filename, *args)
    
    return f'Saved as {filename}', ''

def convert_onnx_to_trt_preset(filename, onnx_filename, *args):
    assert not cmd_opts.disable_extension_access, "won't run the command to create TensorRT file because extension access is dsabled (use --enable-insecure-extension-access)"
    filename = get_trt_filename(filename, onnx_filename)
    export_trt.generate_trt_engine_presets(filename, onnx_filename, *args)
    return f'Saved as {filename}', ''





def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="Convert to ONNX"):
                        gr.HTML(value="<p style='margin-bottom: 0.7em'>Convert currently loaded checkpoint into ONNX. The conversion will fail catastrophically if TensorRT was used at any point prior to conversion, so you might have to restart webui before doing the conversion.</p>")
                        

                        onnx_filename = gr.Textbox(label='Filename', value="", elem_id="onnx_filename", info="Leave empty to use the same name as model and put results into models/Unet-onnx directory")
                        onnx_opset = gr.Number(label='ONNX opset version', precision=0, value=17, info="Leave this alone unless you know what you are doing")

                        button_export_unet = gr.Button(value="Convert Unet to ONNX", variant='primary', elem_id="onnx_export_unet")

                    with gr.Tab(label="Convert ONNX to TensorRT (Preset configs)"):
                        gr.HTML(value="<p style='margin-bottom: 0.7em'>Set configuration to true to build multiple TensorRT engines. (Warning build time increases per selected preset)</p>")
                        
                        with gr.Column(variant="compact"):
                            trt_source_filename_preset = gr.Dropdown(ONNX_MODELS, label="ONNX Model", info="Select an Onnx model to convert to TensorRT. Found in models/Unet-onnx")
                            refresh_onnx_db = gr.Button("Refresh List.")
                        trt_filename = gr.Textbox(label='Output filename', value="", elem_id="trt_filename", info="Leave empty to use the same name as onnx and put results into models/Unet-trt directory", visible=False)
                        
                        with gr.Row():
                            
                            with gr.Column(label="Presets 512x512"):
                                profile_512x512x1 = gr.Checkbox(label='512x512x1', value=False, elem_id="512x512x1")
                                profile_512x512x2 = gr.Checkbox(label='512x512x2', value=False, elem_id="512x512x2")
                                profile_512x512x4 = gr.Checkbox(label='512x512x4', value=False, elem_id="512x512x4")
                                
                            with gr.Column(label="Presets 768x768"):
                                profile_768x768x1 = gr.Checkbox(label='768x768x1', value=False, elem_id="768x768x1")
                                profile_768x768x2 = gr.Checkbox(label='768x768x2', value=False, elem_id="768x768x2")
                                profile_768x768x4 = gr.Checkbox(label='768x768x4', value=False, elem_id="768x768x4")

                        use_fp16 = gr.Checkbox(label='FP16', value=True, elem_id="trt_fp16")
                        button_export_preset_trt = gr.Button(value="Convert ONNX to TensorRT", variant='primary', elem_id="button_export_preset_trt")


                    with gr.Tab(label="Convert ONNX to TensorRT (Custom)"):
                        trt_source_filename = gr.Textbox(label='Onnx model filename', value="", elem_id="trt_source_filename")
                        trt_filename = gr.Textbox(label='Output filename', value="", elem_id="trt_filename", info="Leave empty to use the same name as onnx and put results into models/Unet-trt directory")

                        with gr.Column(elem_id="trt_width"):
                            min_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Minimum width", value=512, elem_id="trt_min_width")
                            opt_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Optimal width", value=512, elem_id="trt_opt_width")
                            max_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Maximum width", value=512, elem_id="trt_max_width")

                        with gr.Column(elem_id="trt_height"):
                            min_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Minimum height", value=512, elem_id="trt_min_height")
                            opt_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Optimal height", value=512, elem_id="trt_opt_height")
                            max_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Maximum height", value=512, elem_id="trt_max_height")

                        with gr.Column(elem_id="trt_batch_size"):
                            min_bs = gr.Slider(minimum=1, maximum=16, step=1, label="Minimum batch size", value=1, elem_id="trt_min_bs")
                            opt_bs = gr.Slider(minimum=1, maximum=16, step=1, label="Optimal batch size", value=1, elem_id="trt_opt_bs")
                            max_bs = gr.Slider(minimum=1, maximum=16, step=1, label="Maximum batch size", value=1, elem_id="trt_max_bs")

                        with gr.Column(elem_id="trt_token_count"):
                            min_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Minimum prompt token count", value=75, elem_id="trt_min_token_count")
                            opt_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Optimal prompt token count", value=75, elem_id="trt_opt_token_count")
                            max_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Maximum prompt token count", value=75, elem_id="trt_max_token_count")

                        trt_extra_args = gr.Textbox(label='Extra arguments', value="", elem_id="trt_extra_args", info="Extra arguments for trtexec command in plain text form")

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            use_fp16 = gr.Checkbox(label='Use half floats', value=True, elem_id="trt_fp16")

                        button_export_trt = gr.Button(value="Convert ONNX to TensorRT", variant='primary', elem_id="trt_convert_from_onnx")

            with gr.Column(variant='panel'):
                trt_result = gr.Label(elem_id="trt_result", value="", show_label=False)
                trt_info = gr.HTML(elem_id="trt_info", value="")


        refresh_onnx_db.click(load_onnx_list,
                              outputs=trt_source_filename_preset)
        button_export_unet.click(
            wrap_gradio_gpu_call(export_unet_to_onnx, extra_outputs=["Conversion failed"]),
            inputs=[onnx_filename, onnx_opset],
            outputs=[trt_result, trt_info],
        )

        button_export_preset_trt.click(
            wrap_gradio_gpu_call(convert_onnx_to_trt_preset, extra_outputs=[""]),
            inputs=[trt_filename, trt_source_filename_preset, profile_512x512x1, profile_512x512x2, profile_512x512x4, profile_768x768x1,profile_768x768x2, profile_768x768x4, use_fp16],
            outputs=[trt_result, trt_info],      
        )
 
        button_export_trt.click(
            wrap_gradio_gpu_call(convert_onnx_to_trt, extra_outputs=[""]),
            inputs=[trt_filename, trt_source_filename, min_bs, opt_bs, max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height, use_fp16, trt_extra_args],
            outputs=[trt_result, trt_info],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]

