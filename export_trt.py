import os.path

from collections import OrderedDict
import os
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine
import torch
import tensorrt as trt

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
        trt_dir = os.path.dirname(self.engine_path)
        if not os.path.exists(trt_dir):
            os.mkdir(trt_dir)

        
        with torch.no_grad():
            torch.cuda.empty_cache()

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

def get_unet_trt_profile(cond_dim, min_bs, opt_bs ,max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height):
    profile = {
            'x' :          ((min_bs * 2, 4, min_height // 8, min_width // 8), (opt_bs * 2, 4, opt_width // 8, opt_height // 8), (max_bs * 2, 4, max_height // 8, max_width // 8)),
            'timesteps' :  ((min_bs * 2,), (opt_bs * 2,), (max_bs * 2,)),
            'context' :    ((min_bs * 2, min_token_count // 75 * 77, cond_dim), (opt_bs * 2,  opt_token_count // 75 * 77, cond_dim), (max_bs * 2, max_token_count // 75 * 77, cond_dim)),
            }
    return profile

def get_trt_profile_filename(profile):
    return "unet_{}x{}x{}".format(int(profile['x'][0][2] * 8), int(profile['x'][0][3] * 8), int(profile['x'][0][0]/2))

def generate_trt_engine_presets(trt_filename, onnx_filename, profile_512_512_1, profile_512_512_4, profile_768x768x1, profile_768x768x4, use_fp16):
    
    cond_dim = 768  # XXX should be detected for SD2.0
    profiles = []
    if profile_512_512_1:
        profiles.append(get_unet_trt_profile(cond_dim, 1, 1 , 1, 75, 75, 75, 512, 512, 512, 512, 512, 512))
    if profile_512_512_4:
        profiles.append(get_unet_trt_profile(cond_dim, 4, 4 , 4, 75, 75, 75, 512, 512, 512, 512, 512, 512))
    if profile_768x768x1:
        profiles.append(get_unet_trt_profile(cond_dim, 1, 1 , 1, 75, 75, 75, 768, 768, 768, 768, 768, 768)) 
    if profile_768x768x4:
        profiles.append(get_unet_trt_profile(cond_dim, 4, 4 , 4, 75, 75, 75, 768, 768, 768, 768, 768, 768))

    for profile in profiles:

        config_name = get_trt_profile_filename(profile)
        trt_engine_name = os.path.dirname(trt_filename) + "\{}.trt".format(config_name)

        if os.path.isfile(trt_engine_name):
            print("Skipping engine build for config: {}, engine already exsists. ({})".format(config_name, trt_engine_name))
            continue

        cache_file = os.path.dirname(trt_filename) + "\{}_timing.cache".format(config_name)
        print("Using cache file {}".format(cache_file))

        builder = EngineBuilder(trt_engine_name)
        builder.build(onnx_path=onnx_filename, fp16=True, input_profile=profile, timing_cache=cache_file)


def generate_trt_engine(trt_filename, onnx_filename, min_bs, opt_bs, max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height, use_fp16, trt_extra_args):
    # cond_dim can be detected with polygraphy
    trt_dir = os.path.dirname(trt_filename)
    cond_dim = 768  # XXX should be detected for SD2.0
    
    profile = get_unet_trt_profile(cond_dim, min_bs, opt_bs, max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height)

    
    config_name = get_trt_profile_filename(profile) 

    
    trt_engine_name = trt_dir + "\{}.trt".format(config_name)
    if os.path.isfile(trt_engine_name):
        print("Skipping engine build for config: {}, engine already exsists. ({})".format(config_name, trt_engine_name))
        return ""
    
    print("Building profile {}".format(profile))
    # find a smart way to detect cache file
    cache_file = os.path.dirname(trt_filename) + "\{}_timing.cache".format(config_name)
    print("Using cache file {}".format(cache_file))

    builder = EngineBuilder(trt_engine_name)
    builder.build(onnx_path=onnx_filename, fp16=use_fp16, input_profile=profile, timing_cache=cache_file)

    return ""
   