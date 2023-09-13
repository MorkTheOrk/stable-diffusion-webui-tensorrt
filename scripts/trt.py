import os
import numpy as np

import ldm.modules.diffusionmodules.openaimodel

import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal

import ui_trt
from utilities import Engine
from exporter import get_cc
from typing import List
from ui_trt import TRTSettings, get_available_trt_unet
from time import time

class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, filename: List[str]):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return TrtUnet(self.filename)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.available_profiles = [TRTSettings.from_hash(p) for p in self.filename]
        self.engine = None
        self.stream = None
        self.controlnet = None

        self.cc_major, self.cc_minor = get_cc()
        self.shape_hash = hash(self.available_profiles[0])

        self.engine = Engine(self.filename[0])
        self.loaded_index = 0

    def forward(self, x, timesteps, context, *args, **kwargs):
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        # Need to check compatability on the fly
        if self.shape_hash != hash(x.shape):
            bs, _, w, h = x.shape
            n_tokens = context.shape[1]
            self.switch_engine(bs, h, w, n_tokens)
            self.shape_hash = hash(x.shape)

        tmp = torch.empty(self.engine_vram_req, dtype=torch.uint8, device=devices.device)
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]
        return out

    def switch_engine(self, bs, h, w, n_tokens):
        valid = [
            (p, i)
            for i, p in enumerate(self.available_profiles)
            if p.is_compatabile(bs, w, h, n_tokens)
        ]
        if len(valid) == 0:
            raise ValueError("No valid profile found")
        distances = [p.distance(bs, w, h) for p, i in valid]
        best = valid[np.argmin(distances)][1]
        if best == self.loaded_index:
            return
        self.deactivate()
        self.engine = Engine(self.filename[best])
        self.activate()
        self.loaded_index = best

    def activate(self):
        self.engine.load()
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

    def deactivate(self):
        del self.engine


TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")

def list_unets(l):
    model = get_available_trt_unet()
    for k, v in model.items():
        l.append(TrtUnetOption(k, v))


script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
