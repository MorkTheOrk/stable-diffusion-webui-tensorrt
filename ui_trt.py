import html
import os

import launch
from modules import script_callbacks, paths_internal, shared
import gradio as gr

from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow

from exporter import export_onnx, export_trt, get_cc
from utilities import PIPELINE_TYPE
from models import make_UNet, make_UNetXL
from logging import info
import logging
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)

ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
    os.makedirs(ONNX_MODEL_DIR)

TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")
if not os.path.exists(TRT_MODEL_DIR):
    os.makedirs(TRT_MODEL_DIR)

ONNX_MODELS = [os.path.join(ONNX_MODEL_DIR, f"{m}") for m in sorted(os.listdir(ONNX_MODEL_DIR))]
TRT_MODELS = [os.path.join(TRT_MODEL_DIR, f"{m}") for m in sorted(os.listdir(TRT_MODEL_DIR))]

NVIDIA_CACHE_URL = "https://"  

def load_onnx_list(): 
    ONNX_MODELS = [ os.path.join(ONNX_MODEL_DIR, f"{m}") for m in sorted(os.listdir(ONNX_MODEL_DIR))]
    return gr.Dropdown.update(choices=ONNX_MODELS)

def get_trt_list():
    def strip_trt(s):
        return "_".join(s.split("_")[:-2])
    TRT_MODELS = [strip_trt(m) for m in TRT_MODELS]
    return gr.Dropdown.update(choices=TRT_MODELS)


def get_version_from_model(sd_model):
    if sd_model.is_sd1:
        return "1.5"
    if sd_model.is_sd2:
        return "2.1"
    if sd_model.is_sdxl:
        return "xl-1.0"
    
@dataclass
class TRTHash:
    trt_max_batch: int
    trt_width: int
    trt_height: int
    trt_token_count: int
    use_fp32: bool
    is_inpaint: bool

    def hash(self) -> int:
        # TODO due to cahnge in final version
        return hex(hash((self.trt_max_batch, self.trt_width, self.trt_height, self.trt_token_count, self.use_fp32, self.is_inpaint)))
      
def get_trt_cache(cc_maj, cc_min, enable_remote=False, force_download=False): # feature disabled  
    cache_name = f"timing_cache_cc{cc_maj}{cc_min}.cache"
    cache_path = os.path.join(TRT_MODEL_DIR, cache_name)
    if not os.path.isfile(cache_path) and enable_remote or force_download: 
        import requests
        download_url = f"{NVIDIA_CACHE_URL}/{cache_name}"
        r = requests.get(url=download_url)
        if r.ok:
            open(cache_path, 'wb').write(r.content)
            return cache_path
        else:
            raise RuntimeWarning("Warning! Could not download remote cache file, falling back.(Status Code {r.status_code})")
    else :
        return cache_path 

def export_unet_to_trt(trt_max_batch, trt_width, trt_height, trt_token_count, use_fp32, is_inpaint, force_export):
    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    cc_major, cc_minor = get_cc()

    trt_max_batch = 1 # TODO
    is_inpaint = False

    trt_option_hash = TRTHash(trt_max_batch, trt_width, trt_height, trt_token_count, use_fp32, is_inpaint).hash()

    if cc_major < 7:
        use_fp32 = True
        info("Disabling FP16 because your GPU does not support it.")
    
    onnx_filename = "_".join([model_name, model_hash]) + ".onnx"
    onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)

    trt_engine_filename = "_".join([model_name, model_hash, f"cc{cc_major}{cc_minor}", trt_option_hash]) + ".trt"
    trt_path = os.path.join(TRT_MODEL_DIR, trt_engine_filename)
    timing_cache = get_trt_cache(cc_major, cc_minor) # TODO try to source from remote

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT
    controlnet = None # TODO Controlnet

    n_tokens = (trt_token_count // 75)

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_UNetXL(version, pipeline, None, "cuda", False, trt_max_batch)
    else:
        modelobj = make_UNet(version, pipeline, None, "cuda", False, trt_max_batch, controlnet)
    
    if not os.path.exists(onnx_path):
        info("No ONNX file found. Exporting...")
        export_onnx(onnx_path, modelobj, profile=modelobj.get_input_profile(1, trt_width, trt_height, False, False))
        info("Exported to ONNX.")

    if not os.path.exists(trt_path) or force_export: # TODO longer token sequences
        info("No TensorRT file found. Building...")
        static_batch = False
        if trt_max_batch == 1:
            static_batch = True
        export_trt(trt_path, onnx_path, timing_cache, profile=modelobj.get_input_profile(1, trt_width, trt_height, static_batch, False), use_fp16=not use_fp32)
        info("Built TensorRT file.")
    else:
        info("TensorRT file found. Skipping build. You can enable Force Export in the Expert settings to force a rebuild.")

    f'Saved as {trt_path}', ''

    
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row().style(equal_height=True):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="Convert to TRT"):
                        gr.HTML(value="<p style='margin-bottom: 0.7em'>Convert currently loaded checkpoint into TensorRT.</p>")
                        
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            is_inpaint = gr.Checkbox(label='Is inpainting Model.', value=False, elem_id="trt_inpaint")

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            is_controlnet = gr.Checkbox(label='Is ControlNet Model.', value=False, elem_id="trt_controlnet")
                        
                        with gr.Accordion("Expert Settings", open=False):
                            with gr.Column(elem_id="trt_width"):
                                trt_width = gr.Slider(minimum=256, maximum=2048, step=64, label="Optimal width", value=512, elem_id="trt_opt_width")

                            with gr.Column(elem_id="trt_height"):
                                trt_height = gr.Slider(minimum=256, maximum=2048, step=64, label="Optimal height", value=512, elem_id="trt_opt_height")

                            with gr.Column(elem_id="trt_max_batch"):
                                trt_max_batch = gr.Slider(minimum=1, maximum=1, step=1, label="Largest batch-size allowed", value=1, elem_id="trt_max_batch") #TODO

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Optimal prompt token count", value=75, elem_id="trt_opt_token_count")

                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                use_fp32 = gr.Checkbox(label='FP32', value=False, elem_id="trt_fp32")

                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                force_rebuild = gr.Checkbox(label='Force Rebuild.', value=False, elem_id="trt_force_rebuild")

                        button_export_unet = gr.Button(value="Convert Unet to TensorRT", variant='primary', elem_id="trt_export_unet", style="color: #76B900; background-color: #76B900;")
                      
            with gr.Column(variant='panel'):
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"), "r") as f:                    
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row().style(equal_height=False):
            with gr.Accordion("Output", open=False):
                trt_result = gr.Label(elem_id="trt_result", value="", show_label=False)
                trt_info = gr.HTML(elem_id="trt_info", value="")

        button_export_unet.click(
            wrap_gradio_gpu_call(export_unet_to_trt, extra_outputs=["Conversion failed"]),
            inputs=[trt_max_batch, trt_width, trt_height, trt_token_count, use_fp32, is_inpaint, force_rebuild],
            outputs=[trt_result, trt_info],
        )

        # button_list_unet.click(
        #     get_trt_list, outputs=trt_source_filename_preset
        # ) #TODO


    return [(trt_interface, "TensorRT", "tensorrt")]

