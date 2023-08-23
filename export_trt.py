import os.path

from collections import OrderedDict
from copy import copy
import numpy as np
import os
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine
import tensorrt as trt
import torch
from cuda import cudart as cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class EngineBuilder():
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
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    # Might be needed in future
    # def refit(self, onnx_path, onnx_refit_path):
    #     def convert_int64(arr):
    #         # TODO: smarter conversion
    #         if len(arr.shape) == 0:
    #             return np.int32(arr)
    #         return arr

    #     def add_to_map(refit_dict, name, values):
    #         if name in refit_dict:
    #             assert refit_dict[name] is None
    #             if values.dtype == np.int64:
    #                 values = convert_int64(values)
    #             refit_dict[name] = values

    #     print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
    #     refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

    #     # Construct mapping from weight names in refit model -> original model
    #     name_map = {}
    #     for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
    #         refit_node = refit_nodes[n]
    #         assert node.op == refit_node.op
    #         # Constant nodes in ONNX do not have inputs but have a constant output
    #         if node.op == "Constant":
    #             name_map[refit_node.outputs[0].name] = node.outputs[0].name
    #         # Handle scale and bias weights
    #         elif node.op == "Conv":
    #             if node.inputs[1].__class__ == gs.Constant:
    #                 name_map[refit_node.name+"_TRTKERNEL"] = node.name+"_TRTKERNEL"
    #             if node.inputs[2].__class__ == gs.Constant:
    #                 name_map[refit_node.name+"_TRTBIAS"] = node.name+"_TRTBIAS"
    #         # For all other nodes: find node inputs that are initializers (gs.Constant)
    #         else:
    #             for i, inp in enumerate(node.inputs):
    #                 if inp.__class__ == gs.Constant:
    #                     name_map[refit_node.inputs[i].name] = inp.name
    #     def map_name(name):
    #         if name in name_map:
    #             return name_map[name]
    #         return name

    #     # Construct refit dictionary
    #     refit_dict = {}
    #     refitter = trt.Refitter(self.engine, TRT_LOGGER)
    #     all_weights = refitter.get_all()
    #     for layer_name, role in zip(all_weights[0], all_weights[1]):
    #         # for speciailized roles, use a unique name in the map:
    #         if role == trt.WeightsRole.KERNEL:
    #             name = layer_name+"_TRTKERNEL"
    #         elif role == trt.WeightsRole.BIAS:
    #             name = layer_name+"_TRTBIAS"
    #         else:
    #             name = layer_name

    #         assert name not in refit_dict, "Found duplicate layer: " + name
    #         refit_dict[name] = None


    #     for n in refit_nodes:
    #         # Constant nodes in ONNX do not have inputs but have a constant output
    #         if n.op == "Constant":
    #             name = map_name(n.outputs[0].name)
    #             print(f"Add Constant {name}\n")
    #             add_to_map(refit_dict, name, n.outputs[0].values)

    #         # Handle scale and bias weights
    #         elif n.op == "Conv":
    #             if n.inputs[1].__class__ == gs.Constant:
    #                 name = map_name(n.name+"_TRTKERNEL")
    #                 add_to_map(refit_dict, name, n.inputs[1].values)

    #             if n.inputs[2].__class__ == gs.Constant:
    #                 name = map_name(n.name+"_TRTBIAS")
    #                 add_to_map(refit_dict, name, n.inputs[2].values)

    #         # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
    #         else:
    #             for inp in n.inputs:
    #                 name = map_name(inp.name)
    #                 if inp.__class__ == gs.Constant:
    #                     add_to_map(refit_dict, name, inp.values)

    #     for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
    #         if weights_role == trt.WeightsRole.KERNEL:
    #             custom_name = layer_name+"_TRTKERNEL"
    #         elif weights_role == trt.WeightsRole.BIAS:
    #             custom_name = layer_name+"_TRTBIAS"
    #         else:
    #             custom_name = layer_name

    #         # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
    #         if layer_name.startswith("onnx::Trilu"):
    #             continue

    #         if refit_dict[custom_name] is not None:
    #             refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
    #         else:
    #             print(f"[W] No refit weights for layer: {layer_name}")

    #     if not refitter.refit_cuda_engine():
    #         print("Failed to refit!")
    #         exit(0)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, workspace_size=0):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs['preview_features'] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs['preview_features'].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)



def get_trt_command(trt_filename, onnx_filename, min_bs, max_bs, min_token_count, max_token_count, min_width, max_width, min_height, max_height, use_fp16, trt_extra_args):

    builder = EngineBuilder(trt_filename)

    # To Automatic1111 cond_dim can be detected with polygraphy
    cond_dim = 768  # XXX should be detected for SD2.0
    import torch
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    # Dict Input tensor name -> min, opt, max
    profile = {'x' :            ((min_bs * 2, 4, min_height // 8, min_width // 8), (min_bs * 2, 4, 512 // 8, 512 // 8), (max_bs * 2, 4, max_height // 8, max_width // 8)),
               'timesteps' :    ((min_bs * 2,), (min_bs * 2,), (max_bs * 2,)),
               'context' :      ((min_bs * 2, min_token_count // 75 * 77, cond_dim), (min_bs * 2,  min_token_count // 75 * 77, cond_dim), (max_bs * 2, max_token_count // 75 * 77, cond_dim)),
               }
    
    print("Building profile {}".fomrat(profile))

    # find a smart way to detect cache file
    cache_file = os.path.dirname(trt_filename) + "\{}".format("trt_timing_cache.cache")
    print("Using cache file {}".format(cache_file))
    builder.build(onnx_path=onnx_filename, fp16=use_fp16, input_profile=profile, timing_cache=cache_file)

    return ""
   