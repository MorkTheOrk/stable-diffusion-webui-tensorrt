import os

from modules import sd_models, shared
import gradio as gr

from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow

from exporter import export_onnx, export_trt
from utilities import PIPELINE_TYPE, Engine
from models import make_OAIUNetXL, make_OAIUNet
import logging
import gc
import torch
from model_manager import modelmanager, cc_major, TRT_MODEL_DIR
from time import sleep

logging.basicConfig(level=logging.INFO)


def get_version_from_model(sd_model):
    if sd_model.is_sd1:
        return "1.5"
    if sd_model.is_sd2:
        return "2.1"
    if sd_model.is_sdxl:
        return "xl-1.0"

class LogLevel:
    Debug = 0
    Info = 1
    Warning = 2
    Error = 3

def log_md(logging_history, message, prefix="**[INFO]:**"):
    logging_history += f"{prefix} {message} \n"
    return logging_history

def export_unet_to_trt(
    batch_min,
    batch_opt,
    batch_max,
    height_min,
    height_opt,
    height_max,
    width_min,
    width_opt,
    width_max,
    token_count_min,
    token_count_opt,
    token_count_max,
    force_export,
    static_shapes,
    controlnet=None,
):
    logging_history = ""

    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        logging_history = log_md(logging_history, "Disabling FP16 because your GPU does not support it.")
        yield logging_history

    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name, model_hash)

    logging_history = log_md(logging_history, f"Exporting {model_name} to TensorRT", prefix="###")
    yield logging_history

    timing_cache = modelmanager.get_timing_cache(allow_remote=False)

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT
    controlnet = None

    min_textlen = (token_count_min // 75) * 77
    opt_textlen = (token_count_opt // 75) * 77
    max_textlen = (token_count_max // 75) * 77

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(
            version, pipeline, "cuda", False, batch_max, opt_textlen, max_textlen
        )
        diable_optimizations = True
    else:
        modelobj = make_OAIUNet(
            version,
            pipeline,
            "cuda",
            False,
            batch_max,
            opt_textlen,
            max_textlen,
            controlnet,
        )
        diable_optimizations = False

    profile = modelobj.get_input_profile(
        batch_min,
        batch_opt,
        batch_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        static_shapes,
    )

    if not os.path.exists(onnx_path):
        logging_history = log_md(logging_history, "No ONNX file found. Exporting...")
        yield logging_history
        export_onnx(
            onnx_path,
            modelobj,
            profile=profile,
            diable_optimizations=diable_optimizations,
        )
        logging_history = log_md(logging_history, "Exported to ONNX.")
        yield logging_history

    trt_engine_filename, trt_path = modelmanager.get_trt_path(
        model_name, model_hash, profile, static_shapes
    )

    if not os.path.exists(trt_path) or force_export:
        logging_history = log_md(logging_history, "No TensorRT file found. Building... This can take a while.")
        yield logging_history
        gc.collect()
        torch.cuda.empty_cache()
        export_trt(
            trt_path,
            onnx_path,
            timing_cache,
            profile=profile,
            use_fp16=not use_fp32,
        )
        logging_history = log_md(logging_history, "Built TensorRT file.")
        yield logging_history
        modelmanager.add_entry(
            model_name,
            model_hash,
            profile,
            static_shapes,
            fp32=use_fp32,
            inpaint=is_inpaint,
            refit=True,
            vram=0,
            unet_hidden_dim=unet_hidden_dim,
            lora=False,
        )  # TODO vram?
    else:
        logging_history = log_md(logging_history, 
            "TensorRT file found. Skipping build. You can enable Force Export in the Expert settings to force a rebuild."
        )
        yield logging_history

    yield logging_history + "\n --- \n ## Exported Successfully \n"


def export_lora_to_trt(lora_name, force_export):
    logging_history = ""
    is_inpaint = False
    use_fp32 = False
    if cc_major < 7:
        use_fp32 = True
        logging_history = log_md(logging_history, "Disabling FP16 because your GPU does not support it.")
        yield logging_history
    unet_hidden_dim = shared.sd_model.model.diffusion_model.in_channels
    if unet_hidden_dim == 9:
        is_inpaint = True

    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    base_name = f"{model_name}_{model_hash}"

    lora_model = available_lora_models[lora_name]

    onnx_base_filename, onnx_base_path = modelmanager.get_onnx_path(
        model_name, model_hash
    )
    onnx_lora_filename, onnx_lora_path = modelmanager.get_onnx_path(
        lora_name, base_name
    )

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT

    if shared.sd_model.is_sdxl:
        pipeline = PIPELINE_TYPE.SD_XL_BASE
        modelobj = make_OAIUNetXL(version, pipeline, "cuda", False, 1, 77, 77)
        diable_optimizations = True
    else:
        modelobj = make_OAIUNet(
            version,
            pipeline,
            "cuda",
            False,
            1,
            77,
            77,
            None,
        )
        diable_optimizations = False

    if not os.path.exists(onnx_lora_path):
        logging_history = log_md(logging_history, "No ONNX file found. Exporting...")
        yield logging_history
        export_onnx(
            onnx_lora_path,
            modelobj,
            profile=modelobj.get_input_profile(
                1, 1, 1, 512, 512, 512, 512, 512, 512, True
            ),
            diable_optimizations=diable_optimizations,
            lora_path=lora_model["filename"],
        )
        logging_history = log_md(logging_history, "Exported to ONNX.")
        yield logging_history

    trt_lora_name = onnx_lora_filename.replace(".onnx", ".trt")
    trt_lora_path = os.path.join(TRT_MODEL_DIR, trt_lora_name)

    available_trt_unet = modelmanager.available_models()
    if len(available_trt_unet[base_name]) == 0:
        logging_history = log_md(logging_history, "Please export the base model first.")
        yield logging_history
    trt_base_path = os.path.join(
        TRT_MODEL_DIR, available_trt_unet[base_name][0]["filepath"]
    )

    if not os.path.exists(onnx_base_path):
        raise ValueError("Please export the base model first.")

    if not os.path.exists(trt_lora_path) or force_export:
        logging_history = log_md(logging_history, "No TensorRT file found. Building...")
        yield logging_history
        engine = Engine(trt_base_path)
        engine.load()
        engine.refit(onnx_base_path, onnx_lora_path, dump_refit_path=trt_lora_path)
        logging_history = log_md(logging_history, "Built TensorRT file.")
        yield logging_history

        modelmanager.add_lora_entry(
            base_name,
            lora_name,
            trt_lora_name,
            use_fp32,
            is_inpaint,
            0,
            unet_hidden_dim,
        )
    yield logging_history + "\n --- \n ## Exported Successfully \n"


profile_presets = {
    "512x512 (Static)": (1, 1, 1, 512, 512, 512, 512, 512, 512, 75, 75, 75),
    "768x768 (Static)": (1, 1, 1, 768, 768, 768, 768, 768, 768, 75, 75, 75),
    "1024x1024 (Static)": (1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75),
    "512x512 (Dynamic)": (1, 1, 1, 256, 512, 512, 256, 512, 512, 75, 75, 150),
    "768x768 (Dynamic)": (1, 1, 1, 512, 768, 1024, 512, 768, 1024, 75, 75, 150),
    "1024x1024 (Dynamic)": (1, 1, 1, 512, 1024, 1536, 512, 1024, 1536, 75, 75, 150),
}


def get_settings_from_version(version):
    static = False
    if "Static" in version:
        static = True
    return *profile_presets[version], static


def diable_export(version):
    if version is None:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)


def diable_visibility(hide):
    num_outputs = 8
    out = [gr.update(visible=not hide) for _ in range(num_outputs)]
    return out


def engine_profile_card():
    available_models = modelmanager.available_models()
    out_string = "## Available TensorRT Models \n"
    for base_model, models in available_models.items():
        markdown_string = f"### Model: {base_model} \n"
        for i, m in enumerate(models):
            if m["config"].lora:
                continue
            markdown_string += f"#### Profile {i} \n"
            for dim, profile in m["config"].profile.items():
                markdown_string += " - **{}:** {} \n".format(dim, str(profile))
            markdown_string += "\n"
        out_string += markdown_string
        out_string += "\n --- \n"
    # print(out_string)
    return out_string


def get_version_from_filename(name):
    if "v1-" in name:
        return "1.5"
    elif "v2-" in name:
        return "2.1"
    elif "xl" in name:
        return "xl-1.0"
    else:
        return "Unknown"


available_lora_models = {}


def get_lora_checkpoints():
    canditates = list(
        shared.walk_files(
            shared.cmd_opts.lora_dir,
            allowed_extensions=[".pt", ".ckpt", ".safetensors"],
        )
    )
    for filename in canditates:
        name = os.path.splitext(os.path.basename(filename))[0]
        metadata = sd_models.read_metadata_from_safetensors(filename)
        available_lora_models[name] = {
            "filename": filename,
            "version": get_version_from_filename(metadata["ss_sd_model_name"]),
        }
    return [
        k
        for k, v in available_lora_models.items()
        if v["version"] == get_version_from_model(shared.sd_model)
    ]

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="TensorRT Exporter"):
                        gr.Markdown(
                            value="# TensorRT Exporter",
                        )

                        version = gr.Dropdown(
                            label="Preset",
                            choices=list(profile_presets.keys()),
                            elem_id="sd_version",
                            default=None,
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                static_shapes = gr.Checkbox(
                                    label="Use static shapes.",
                                    value=True,
                                    elem_id="trt_static_shapes",
                                )

                            with gr.Column(elem_id="trt_width"):
                                trt_width_opt = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Optimal width",
                                    value=512,
                                    elem_id="trt_opt_width",
                                )
                                trt_width_min = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Min width",
                                    value=256,
                                    elem_id="trt_min_width",
                                )
                                trt_width_max = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Max width",
                                    value=768,
                                    elem_id="trt_max_width",
                                )

                            with gr.Column(elem_id="trt_height"):
                                trt_height_opt = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Optimal height",
                                    value=512,
                                    elem_id="trt_opt_height",
                                )
                                trt_height_min = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Min height",
                                    value=256,
                                    elem_id="trt_min_height",
                                )
                                trt_height_max = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    label="Max height",
                                    value=768,
                                    elem_id="trt_max_height",
                                )

                            with gr.Column(elem_id="trt_max_batch"):
                                trt_opt_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Optimal batch-size",
                                    value=1,
                                    elem_id="trt_opt_batch",
                                )
                                trt_min_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Min batch-size",
                                    value=1,
                                    elem_id="trt_min_batch",
                                )
                                trt_max_batch = gr.Slider(
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    label="Max batch-size",
                                    value=1,
                                    elem_id="trt_max_batch",
                                )

                            with gr.Column(elem_id="trt_token_count"):
                                trt_token_count_opt = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Optimal prompt token count",
                                    value=75,
                                    elem_id="trt_opt_token_count_opt",
                                )
                                trt_token_count_min = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Min prompt token count",
                                    value=75,
                                    elem_id="trt_opt_token_count_min",
                                )
                                trt_token_count_max = gr.Slider(
                                    minimum=75,
                                    maximum=750,
                                    step=75,
                                    label="Max prompt token count",
                                    value=150,
                                    elem_id="trt_opt_token_count_max",
                                )

                            with FormRow(
                                elem_classes="checkboxes-row", variant="compact"
                            ):
                                force_rebuild = gr.Checkbox(
                                    label="Force Rebuild.",
                                    value=False,
                                    elem_id="trt_force_rebuild",
                                )

                        button_export_unet = gr.Button(
                            value="Convert Unet to TensorRT",
                            variant="primary",
                            elem_id="trt_export_unet",
                            visible=False,
                        )

                        version.change(
                            get_settings_from_version,
                            version,
                            [
                                trt_min_batch,
                                trt_opt_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_opt,
                                trt_height_max,
                                trt_width_min,
                                trt_width_opt,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_opt,
                                trt_token_count_max,
                                static_shapes,
                            ],
                        )
                        version.change(diable_export, version, button_export_unet)
                        static_shapes.change(
                            diable_visibility,
                            static_shapes,
                            [
                                trt_min_batch,
                                trt_max_batch,
                                trt_height_min,
                                trt_height_max,
                                trt_width_min,
                                trt_width_max,
                                trt_token_count_min,
                                trt_token_count_max,
                            ],
                        )

                    with gr.Tab(label="TensorRT LoRA"):
                        gr.Markdown("# Apply LoRA checkpoint to TensorRT model")
                        lora_refresh_button = gr.Button(
                            value="Refresh",
                            variant="primary",
                            elem_id="trt_lora_refresh",
                        )

                        trt_lora_dropdown = gr.Dropdown(
                            choices=get_lora_checkpoints(),
                            elem_id="lora_model",
                            label="LoRA Model",
                            default=None,
                        )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            trt_lora_force_rebuild = gr.Checkbox(
                                label="Force Rebuild.",
                                value=False,
                                elem_id="trt_lora_force_rebuild",
                            )

                        button_export_lora_unet = gr.Button(
                            value="Convert to TensorRT",
                            variant="primary",
                            elem_id="trt_lora_export_unet",
                            visible=False,
                        )

                        lora_refresh_button.click(
                            get_lora_checkpoints,
                            None,
                            trt_lora_dropdown,
                        )
                        trt_lora_dropdown.change(
                            diable_export, trt_lora_dropdown, button_export_lora_unet
                        )

            with gr.Column(variant="panel"):
                with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"),
                    "r",
                ) as f:
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=True):
                trt_result = gr.Markdown(elem_id="trt_result", value="")

        with gr.Row(equal_height=False):
            trt_available_models = gr.Markdown(
                elem_id="trt_available_models", value=engine_profile_card()
            )

        button_export_unet.click(
            export_unet_to_trt,
            inputs=[
                trt_min_batch,
                trt_opt_batch,
                trt_max_batch,
                trt_height_min,
                trt_height_opt,
                trt_height_max,
                trt_width_min,
                trt_width_opt,
                trt_width_max,
                trt_token_count_min,
                trt_token_count_opt,
                trt_token_count_max,
                force_rebuild,
                static_shapes,
            ],
            outputs=[trt_result],
        )

        button_export_lora_unet.click(
            wrap_gradio_gpu_call(
                export_lora_to_trt, extra_outputs=["Conversion failed"]
            ),
            inputs=[trt_lora_dropdown, trt_lora_force_rebuild],
            outputs=[trt_result],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]
