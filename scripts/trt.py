import os
import numpy as np

import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal

import ui_trt
from cuda import cudart as cudart

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return TrtUnet(self.filename)


np_to_torch = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int32: torch.int32,
}


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.engine = None
        self.trtcontext = None
        self.buffers = None
        self.buffers_shape = ()
        self.nptype = None
        self.cuda_graph_instance = None

    def allocate_buffers(self, feed_dict):
        from tensorrt import Dims
        buffers_shape = sum([x.shape for x in feed_dict.values()], ())
        if self.buffers_shape == buffers_shape:
            return

        self.buffers_shape = buffers_shape
        self.buffers = {}

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = self.nptype(self.engine.get_binding_dtype(binding))

            if binding in feed_dict:
                shape = Dims(feed_dict[binding].shape)
            else:
                shape = self.trtcontext.get_binding_shape(binding_idx)

            if self.engine.binding_is_input(binding):
                if not self.trtcontext.set_binding_shape(binding_idx, shape): # return value might be borked
                    print(f'bad shape for TensorRT input {binding}: {tuple(shape)}')

            tensor = torch.empty(tuple(shape), dtype=np_to_torch[dtype], device=devices.device)
            self.buffers[binding] = tensor

    def infer(self, feed_dict):
        
        if not self.buffers:
            self.allocate_buffers(feed_dict)

        for name, tensor in feed_dict.items():
            self.buffers[name].copy_(tensor)

        for name, tensor in self.buffers.items():
            self.trtcontext.set_tensor_address(name, tensor.data_ptr())

        if self.cuda_graph_instance is not None:
            if cudart.cudaGraphLaunch(self.cuda_graph_instance, self.cudaStream)[0] != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Error in trt.py: Could not run cudaGraph")
        else:
            # do inference before CUDA graph capture
            self.trtcontext.execute_async_v3(self.cudaStream)
            # capture cuda graph
            CUASSERT(cudart.cudaStreamBeginCapture(self.cudaStream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
            self.trtcontext.execute_async_v3(self.cudaStream)
            self.graph = CUASSERT(cudart.cudaStreamEndCapture(self.cudaStream))
            self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiateWithFlags(self.graph, 0))

    def forward(self, x, timesteps, context, *args, **kwargs):
        self.infer({"x": x, "timesteps": timesteps, "context": context})
        return self.buffers["output"].to(dtype=x.dtype, device=devices.device)

    def activate(self):
        import tensorrt as trt  # we import this late because it breaks torch onnx export

        TRT_LOGGER = trt.Logger(trt.ILogger.Severity.VERBOSE)
        trt.init_libnvinfer_plugins(None, "")
        self.nptype = trt.nptype

        with open(self.filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.trtcontext = self.engine.create_execution_context()
        self.cudaStream = cudart.cudaStreamCreate()[1]
        assert(self.trtcontext)

    def deactivate(self):
        self.engine = None
        self.trtcontext = None
        self.buffers = None
        self.buffers_shape = ()
        devices.torch_gc()


def list_unets(l):

    trt_dir = os.path.join(paths_internal.models_path, 'Unet-trt')
    candidates = list(shared.walk_files(trt_dir, allowed_extensions=[".trt"]))
    for filename in sorted(candidates, key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]

        opt = TrtUnetOption(filename, name)
        l.append(opt)


script_callbacks.on_list_unets(list_unets)

script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
