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
from models import make_OAIUNetXL, make_OAIUNet
from logging import info
import logging
from dataclasses import dataclass
import gc
import torch
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
if not os.path.exists(ONNX_MODEL_DIR):
    os.makedirs(ONNX_MODEL_DIR)

TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")
if not os.path.exists(TRT_MODEL_DIR):
    os.makedirs(TRT_MODEL_DIR)

NVIDIA_CACHE_URL = ""


def get_version_from_model(sd_model):
    if sd_model.is_sd1:
        return "1.5"
    if sd_model.is_sd2:
        return "2.1"
    if sd_model.is_sdxl:
        return "xl-1.0"


@dataclass
class TRTSettings:
    trt_batch_min: int
    trt_batch_opt: int
    trt_batch_max: int
    trt_height_min: int
    trt_height_opt: int
    trt_height_max: int
    trt_width_min: int
    trt_width_opt: int
    trt_width_max: int
    trt_token_count_min: int
    trt_token_count_opt: int
    trt_token_count_max: int
    use_fp32: bool
    is_static_shape: bool
    unet_hidden_dim: int = 4

    def hash(self) -> int:
        _opt = "x".join(
            [
                str(self.trt_batch_opt),
                str(self.trt_height_opt),
                str(self.trt_width_opt),
                str(self.trt_token_count_opt),
            ]
        )
        out = [_opt]
        if not self.is_static_shape:
            _min = "x".join(
                [
                    str(self.trt_batch_min),
                    str(self.trt_height_min),
                    str(self.trt_width_min),
                    str(self.trt_token_count_min),
                ]
            )
            out.append(_min)
            _max = "x".join(
                [
                    str(self.trt_batch_max),
                    str(self.trt_height_max),
                    str(self.trt_width_max),
                    str(self.trt_token_count_max),
                ]
            )
            out.append(_max)

        if self.use_fp32:
            out.append("fp32")
        return "-".join(out)

    def is_compatabile(self, bs, w, h, n_tokens):
        if self.is_static_shape:
            return (
                bs == self.trt_batch_opt * 2
                and w == self.trt_width_opt // 8
                and h == self.trt_height_opt // 8
                and n_tokens == self.trt_token_count_opt
            )
        else:
            return (
                bs >= self.trt_batch_min
                and bs <= self.trt_batch_max * 2
                and w >= self.trt_width_min // 8
                and w <= self.trt_width_max // 8
                and h >= self.trt_height_min // 8
                and h <= self.trt_height_max // 8
                and n_tokens >= self.trt_token_count_min
                and n_tokens <= self.trt_token_count_max
            )

    def distance(self, bs, h, w):
        return (
            abs(bs - self.trt_batch_opt * 2)
            + abs(h - self.trt_height_opt // 8)
            + abs(w - self.trt_width_opt // 8)
            + abs(w - self.trt_width_max // 8)
            + abs(h - self.trt_height_max // 8)
            + abs(bs - self.trt_batch_max * 2)
        )

    @staticmethod
    def from_hash(hash: str):
        configs = hash.split("_")[-1].split(".")[0]
        configs = configs.split("-")
        fp32 = False
        static_shape = False
        if len(configs) == 1:
            _opt = configs[0].split("x")
            _min = configs[0].split("x")
            _max = configs[0].split("x")
            static_shape = True
        if len(configs) >= 3:
            _opt, _min, _max = (
                configs[0].split("x"),
                configs[1].split("x"),
                configs[2].split("x"),
            )
            if len(configs) == 4:
                fp32 = True

        return TRTSettings(
            trt_batch_min=int(_min[0]),
            trt_batch_opt=int(_opt[0]),
            trt_batch_max=int(_max[0]),
            trt_height_min=int(_min[1]),
            trt_height_opt=int(_opt[1]),
            trt_height_max=int(_max[1]),
            trt_width_min=int(_min[2]),
            trt_width_opt=int(_opt[2]),
            trt_width_max=int(_max[2]),
            trt_token_count_min=(int(_min[3])//75)*77,
            trt_token_count_opt=(int(_opt[3])//75)*77,
            trt_token_count_max=(int(_max[3])//75)*77,
            use_fp32=fp32,
            is_static_shape=static_shape,
        )

    def __hash__(self) -> int:
        return hash(
            torch.Size(
                [
                    self.trt_batch_opt * 2,
                    self.unet_hidden_dim,
                    self.trt_height_opt // 8,
                    self.trt_width_opt // 8,
                ]
            )
        )


def get_trt_cache(
    cc_maj, cc_min, enable_remote=False, force_download=False
):  # feature disabled
    cache_name = f"timing_cache_cc{cc_maj}{cc_min}.cache"
    cache_path = os.path.join(TRT_MODEL_DIR, cache_name)
    if not os.path.isfile(cache_path) and enable_remote or force_download:
        import requests

        download_url = f"{NVIDIA_CACHE_URL}/{cache_name}"
        r = requests.get(url=download_url)
        if r.ok:
            open(cache_path, "wb").write(r.content)
            return cache_path
        else:
            raise RuntimeWarning(
                "Warning! Could not download remote cache file, falling back.(Status Code {r.status_code})"
            )
    else:
        return cache_path


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
    use_fp32,
    force_export,
    static_shapes,
    controlnet,
):
    model_hash = shared.sd_model.sd_checkpoint_info.hash
    model_name = shared.sd_model.sd_checkpoint_info.model_name
    cc_major, cc_minor = get_cc()
    is_inpaint = False

    trt_settings = TRTSettings(
        trt_batch_min=batch_min if not static_shapes else batch_opt,
        trt_batch_opt=batch_opt,
        trt_batch_max=batch_max if not static_shapes else batch_opt,
        trt_height_min=height_min if not static_shapes else height_opt,
        trt_height_opt=height_opt,
        trt_height_max=height_max if not static_shapes else height_opt,
        trt_width_min=width_min if not static_shapes else width_opt,
        trt_width_opt=width_opt,
        trt_width_max=width_max if not static_shapes else width_opt,
        trt_token_count_min=token_count_min if not static_shapes else token_count_opt,
        trt_token_count_opt=token_count_opt,
        trt_token_count_max=token_count_max if not static_shapes else token_count_opt,
        use_fp32=use_fp32,
        is_static_shape=static_shapes,
        unet_hidden_dim=shared.sd_model.model.diffusion_model.in_channels,
    )
    trt_option_hash = trt_settings.hash()

    if trt_settings.unet_hidden_dim == 9:
        is_inpaint = True

    if cc_major < 7:
        use_fp32 = True
        info("Disabling FP16 because your GPU does not support it.")

    onnx_filename = "_".join([model_name, model_hash]) + ".onnx"
    onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)

    trt_engine_filename = (
        "_".join([model_name, model_hash, f"cc{cc_major}{cc_minor}", trt_option_hash])
        + ".trt"
    )
    trt_path = os.path.join(TRT_MODEL_DIR, trt_engine_filename)
    timing_cache = get_trt_cache(cc_major, cc_minor)

    version = get_version_from_model(shared.sd_model)

    pipeline = PIPELINE_TYPE.TXT2IMG
    if is_inpaint:
        pipeline = PIPELINE_TYPE.INPAINT
    controlnet = None  # TODO Controlnet

    min_textlen = (trt_settings.trt_token_count_max // 75) * 77
    opt_textlen = (trt_settings.trt_token_count_opt // 75) * 77
    max_textlen = (trt_settings.trt_token_count_max // 75) * 77

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

    if not os.path.exists(onnx_path):
        info("No ONNX file found. Exporting...")
        export_onnx(
            onnx_path,
            modelobj,
            profile=modelobj.get_input_profile(1, width_opt, height_opt, False, False),
            diable_optimizations=diable_optimizations,
        )
        info("Exported to ONNX.")

    if not os.path.exists(trt_path) or force_export:
        info("No TensorRT file found. Building...")
        static_batch = False
        if batch_max == batch_min:
            static_batch = True
        gc.collect()
        torch.cuda.empty_cache()
        export_trt(
            trt_path,
            onnx_path,
            timing_cache,
            profile=modelobj.get_input_profile(
                1, width_opt, height_opt, static_batch, static_shapes
            ),
            use_fp16=not use_fp32,
        )
        info("Built TensorRT file.")
    else:
        info(
            "TensorRT file found. Skipping build. You can enable Force Export in the Expert settings to force a rebuild."
        )

    f"Saved as {trt_path}", ""


def get_settings_from_version(version):
    if version == "1.x":
        return 1, 1, 1, 256, 512, 512, 256, 512, 512, 75, 75, 150
    elif version == "2.x":
        return 1, 1, 1, 256, 512, 768, 256, 512, 768, 75, 75, 150
    elif version == "XL Base":
        return 1, 1, 1, 512, 1024, 1024, 512, 1024, 1024, 75, 75, 150
    else:
        return 1, 1, 1, 256, 512, 512, 256, 512, 512, 75, 75, 150


def diable_export(version):
    if version is None:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)


def diable_visibility(hide):
    num_outputs = 8
    out = [gr.update(visible=not hide) for _ in range(num_outputs)]
    return out


def get_available_trt_unet():
    a, b = get_cc()
    model = defaultdict(list)
    for p in os.listdir(TRT_MODEL_DIR):
        base_model = "_".join(p.split("_")[:-2])
        if f"_cc{a}{b}" not in p:
            continue
        if not p.endswith(".trt"):
            continue
        model[base_model].append(os.path.join(TRT_MODEL_DIR, p))
    return model


def engine_profile_card():
    models = get_available_trt_unet()
    out_string = "## Available TensorRT Models \n"
    for k, v in models.items():
        markdown_string = f"### Model: {k} \n"
        for i, p in enumerate(v):
            profile = TRTSettings.from_hash(p)
            markdown_string += f"#### Profile {i} \n"
            markdown_string += f" - **Opt:** {profile.trt_batch_opt}x{profile.trt_height_opt}x{profile.trt_width_opt}x{profile.trt_token_count_opt} \n"
            if profile.is_static_shape:
                continue
            else:
                min_max = f" - **Min:** {profile.trt_batch_min}x{profile.trt_height_min}x{profile.trt_width_min}x{profile.trt_token_count_min} \n - **Max:** {profile.trt_batch_max}x{profile.trt_height_max}x{profile.trt_width_max}x{profile.trt_token_count_max} \n"
                markdown_string += min_max
            markdown_string += "\n"
        out_string += markdown_string
        out_string += "\n --- \n"
    # print(out_string)
    return out_string


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                gr.Markdown(
                    value="# TensorRT Exporter",
                )

                version = gr.Dropdown(
                    label="Stable Diffusion Version",
                    choices=["1.x", "2.x", "XL Base"],
                    elem_id="sd_version",
                    default=None,
                )

                with FormRow(elem_classes="checkboxes-row", variant="compact"):
                    is_controlnet = gr.Checkbox(
                        label="Is ControlNet Model.",
                        value=False,
                        elem_id="trt_controlnet",
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
                        static_shapes = gr.Checkbox(
                            label="Use static shapes.",
                            value=False,
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

                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
                        use_fp32 = gr.Checkbox(
                            label="FP32", value=False, elem_id="trt_fp32"
                        )

                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
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

            with gr.Column(variant="panel"):
                with open(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.md"),
                    "r",
                ) as f:
                    trt_info = gr.Markdown(elem_id="trt_info", value=f.read())

        with gr.Row(equal_height=False):
            with gr.Accordion("Output", open=False):
                trt_result = gr.Label(elem_id="trt_result", value="", show_label=False)
                trt_info = gr.HTML(elem_id="trt_info", value="")

        with gr.Row(equal_height=False):
            trt_available_models = gr.Markdown(
                elem_id="trt_available_models", value=engine_profile_card()
            )

        button_export_unet.click(
            wrap_gradio_gpu_call(
                export_unet_to_trt, extra_outputs=["Conversion failed"]
            ),
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
                use_fp32,
                force_rebuild,
                static_shapes,
                is_controlnet,
            ],
            outputs=[trt_result, trt_info],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]
