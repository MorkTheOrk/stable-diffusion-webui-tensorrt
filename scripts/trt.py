import os
import numpy as np

import ldm.modules.diffusionmodules.openaimodel

import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal

import ui_trt
from utilities import Engine
from exporter import get_cc

class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return TrtUnet(self.filename)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.engine = None
        self.trtcontext = None
        self.buffers = None
        self.buffers_shape = {}
        self.nptype = None
        self.cuda_graph_instance = None

        self.engine = None
        self.stream = None
        self.controlnet = None
        
        self.cc_major, self.cc_minor = get_cc()
        self.active = False
        
    def get_shape_dict(self, batch_size, unet_dim, latent_height, latent_width, text_maxlen, embedding_dim, image_height=None, image_width=None):
        if self.controlnet is None:
            return {
                'sample': (batch_size, unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (batch_size, text_maxlen, embedding_dim),
                'latent': (batch_size, 4, latent_height, latent_width)
            }
        else:
            return {
                'sample': (batch_size, unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (batch_size, text_maxlen, embedding_dim),
                'images': (len(self.controlnet), batch_size, 3, image_height, image_width), 
                'latent': (batch_size, 4, latent_height, latent_width)
            }

    def forward(self, x, timesteps, context, *args, **kwargs):
        feed_dict = {
            "sample": x,
            "timestep": timesteps[:1].clone(),
            "encoder_hidden_states": context,
        }

        # tmp = torch.empty(self.engine_vram_req, dtype=torch.uint8, device=devices.device)
        # self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]
        return out

    def activate(self):
        self.engine = Engine(self.filename)
        self.engine.load()
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(False)

    def deactivate(self):
        del self.engine

TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")

def list_unets(l):
    def strip_trt(s):
        return "_".join(s.split("_")[:-2])
    a, b = get_cc()
    trt_models = [m for m in sorted(os.listdir(TRT_MODEL_DIR)) if not "timing_cache" in m]
    model_names = [strip_trt(m) for m in trt_models]
    for p, name in zip(trt_models, model_names):
        if f"_cc{a}{b}" not in p:
            continue
        l.append(TrtUnetOption(os.path.join(TRT_MODEL_DIR, p), name))

script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
