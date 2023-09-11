import torch
import torch.nn.functional as F
import onnx
from logging import info
import time

from modules import sd_hijack, sd_unet
from modules import shared, devices

from utilities import Engine
import os

def get_cc():
    cc_major = torch.cuda.get_device_properties(0).major
    cc_minor = torch.cuda.get_device_properties(0).minor
    return cc_major, cc_minor

def export_onnx(onnx_path, modelobj=None, profile=None, opset=17):
    swap_sdpa = hasattr(F, "scaled_dot_product_attention")
    old_sdpa = getattr(F, "scaled_dot_product_attention", None) if swap_sdpa else None
    if swap_sdpa:
        delattr(F, "scaled_dot_product_attention")

    def disable_checkpoint(self):
        if getattr(self, 'use_checkpoint', False) == True:
            self.use_checkpoint = False
        if getattr(self, 'checkpoint', False) == True:
            self.checkpoint = False

    shared.sd_model.model.diffusion_model.apply(disable_checkpoint)

    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.apply_optimizations('None')

    info("Exporting to ONNX...")
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = modelobj.get_sample_input(
            profile["sample"][1][0] // 2,
            profile["sample"][1][-2] * 8,
            profile["sample"][1][-1] * 8,
        )
        model = shared.sd_model.model.diffusion_model
        torch.onnx.export(
            model,
            inputs,
            "tmp.onnx",
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=modelobj.get_input_names(),
            output_names=modelobj.get_output_names(),
            dynamic_axes=modelobj.get_dynamic_axes(),
        )

    info("Optimize ONNX.")

    onnx_graph = onnx.load("tmp.onnx")
    onnx_opt_graph = modelobj.optimize(onnx_graph)

    if onnx_opt_graph.ByteSize() > 2147483648:
        onnx.save_model(
            onnx_opt_graph,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )
    else:
        onnx.save(onnx_opt_graph, onnx_path)
    info("ONNX export complete.")
    
    # CleanUp
    del onnx_opt_graph
    if swap_sdpa and old_sdpa:
        setattr(F, "scaled_dot_product_attention", old_sdpa)
    sd_hijack.model_hijack.apply_optimizations()
    sd_unet.apply_unet()
    os.remove("tmp.onnx")
    del model


def export_trt(trt_path, onnx_path, timing_cache, profile, use_fp16):
    engine = Engine(trt_path)
    s = time.time()
    engine.build(
            onnx_path,
            use_fp16,
            enable_refit=True,
            enable_preview=True,
            timing_cache=timing_cache,
            input_profile=[profile],
            # hwCompatibility=hwCompatibility,
        )
    e = time.time()
    info(f"Time taken to build: {(e-s)}s")
