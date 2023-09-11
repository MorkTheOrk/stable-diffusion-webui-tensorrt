#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
import tensorrt as trt
import torch
from enum import Enum, auto

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

class PIPELINE_TYPE(Enum):
    TXT2IMG = auto()
    IMG2IMG = auto()
    INPAINT = auto()
    SD_XL_BASE = auto()
    SD_XL_REFINER = auto()

    def is_txt2img(self):
        return self == self.TXT2IMG

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_inpaint(self):
        return self == self.INPAINT

    def is_sd_xl_base(self):
        return self == self.SD_XL_BASE

    def is_sd_xl_refiner(self):
        return self == self.SD_XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()

class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTKERNEL"] = node.name+"_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTBIAS"] = node.name+"_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name
        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name+"_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name+"_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None


        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name+"_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name+"_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, update_output_names=None):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=p,
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding].shape
            else:
                shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # if use_cuda_graph:
        #     if self.cuda_graph_instance is not None:
        #         CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
        #         CUASSERT(cudart.cudaStreamSynchronize(stream))
        #     else:
        #         # do inference before CUDA graph capture
        #         noerror = self.context.execute_async_v3(stream)
        #         if not noerror:
        #             raise ValueError(f"ERROR: inference failed.")
        #         # capture cuda graph
        #         CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
        #         self.context.execute_async_v3(stream)
        #         self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
        #         self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        # else:
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors

    def __str__(self):
        out = ""
        for opt_profile in range(self.engine.num_optimization_profiles):
            out += f"Profile {opt_profile}:\n"
            for binding_idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(binding_idx) 
                shape = self.engine.get_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out