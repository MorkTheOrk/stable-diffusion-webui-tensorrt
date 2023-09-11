import os.path

from collections import OrderedDict
import os
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine

from polygraphy.tools.base import Tool
import torch
import tensorrt as trt
from polygraphy.backend.onnx import onnx_from_path
import polygraphy.backend.onnx.util as onnx_util

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
# Copied from TensorRT GitHub and modified
# https://github.com/NVIDIA/TensorRT/blob/35477bdb94eab72862ffbdf66d4419e408bef45f/demo/Diffusion/utilities.py#L71



class EngineBuilder():
    TRT_REMOTE_CACHE_FILE = None
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None

    def __del__(self):
        del self.engine
        del self.context

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, workspace_size=0):
        trt_dir = os.path.dirname(self.engine_path)

        os.makedirs(trt_dir, exist_ok=True)
        
        torch.cuda.empty_cache()


        print(f"Building TensorRT engine for ONNX {onnx_path} : {self.engine_path}")

        replaceList = ['"']
        for match in replaceList:
            if match in onnx_path:
                 onnx_path = onnx_path.replace(match,'')
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

        
NVIDIA_CACHE_URL = "https://"  
def fetch_remote_cache(sd_version, sd_hash, trt_path):
    if EngineBuilder.TRT_REMOTE_CACHE_FILE is None or False: # feature disabled
        import requests
        cc_maj, cc_min = torch.cuda.get_device_capability()
        remote_cache_name = f"{sd_version}_{sd_hash}_cc_{cc_maj}{cc_min}.cache"
        download_url = f"{NVIDIA_CACHE_URL}"
        r = requests.get(url=download_url)
        if r.ok:
            r_cache_path = os.path.join(trt_path, remote_cache_name)
            open(r_cache_path, 'wb').write(r.content)
            EngineBuilder.TRT_REMOTE_CACHE_FILE = r_cache_path
        else:
            raise RuntimeWarning("Warning! Could not download remote cache file, falling back.(Status Code {r.status_code})")


        
    

def get_unet_trt_profile(cond_dim, min_bs, opt_bs ,max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height):
    profile = {
            'x' :          ((min_bs * 2, 4, min_height // 8, min_width // 8), (opt_bs * 2, 4, opt_width // 8, opt_height // 8), (max_bs * 2, 4, max_height // 8, max_width // 8)),
            'timesteps' :  ((min_bs * 2,), (opt_bs * 2,), (max_bs * 2,)),
            'context' :    ((min_bs * 2, min_token_count // 75 * 77, cond_dim), (opt_bs * 2,  opt_token_count // 75 * 77, cond_dim), (max_bs * 2, max_token_count // 75 * 77, cond_dim)),
            }
    return profile



def get_trt_profile_filename(profile):
    return f"unet_\
    w[{int(profile['x'][0][3] * 8)},{int(profile['x'][1][3] * 8)},{int(profile['x'][2][3] * 8)}], \
    h[{int(profile['x'][0][2] * 8),},{int(profile['x'][1][2] * 8)},{int(profile['x'][2][2] * 8)}],\
    b[{int(profile['x'][0][0]/2)},{int(profile['x'][1][0]/2)},{int(profile['x'][2][0]/2)}], \
    t[{int(profile['context'][0][1]/77*75)},{int(profile['context'][1][1]/77*75)},{int(profile['context'][2][1]/77*75)}]"

def get_cond_dim_from_onnx(onnx_file):
        cond_dim = onnx_util.str_from_onnx(onnx_from_path(onnx_file, ignore_external_data=True)).strip().split("'sequence_length', ")[1].split(")]}\n\n")[0]
        return int(cond_dim)

def generate_trt_engine_presets(trt_filename, onnx_filename, profile_512x512x1, profile_512x512x2, profile_512x512x4, profile_768x768x1, profile_768x768x2, profile_768x768x4, use_fp16):
    
    trt_dir = os.path.dirname(trt_filename)
    cond_dim = get_cond_dim_from_onnx(onnx_filename)
    fetch_remote_cache(8,9,trt_dir)
    profiles = {}
    if profile_512x512x1:
        profiles["512x512x1"] = get_unet_trt_profile(cond_dim, 1, 1 ,1, 77, 154, 154, 512, 512, 512, 512, 512, 512)
    if profile_512x512x2:
        profiles["512x512x2"] = get_unet_trt_profile(cond_dim, 2, 2 ,2, 77, 154, 154, 512, 512, 512, 512, 512, 512)   
    if profile_512x512x4:        
        profiles["512x512x4"] = get_unet_trt_profile(cond_dim, 4, 4 ,4, 77, 154, 154, 512, 512, 512, 512, 512, 512)
    if profile_768x768x1:        
        profiles["768x768x1"] = get_unet_trt_profile(cond_dim, 1, 1 ,1, 77, 154, 154, 768, 768, 768, 768, 768, 768) 
    if profile_768x768x2:        
        profiles["512x512x2"] = get_unet_trt_profile(cond_dim, 2, 2 ,2, 77, 154, 154, 768, 768, 768, 768, 768, 768)   
    if profile_768x768x4:        
        profiles["512x512x4"] = get_unet_trt_profile(cond_dim, 4, 4 ,4, 77, 154, 154, 768, 768, 768, 768, 768, 768)

 
    
    for name, profile in profiles.items():

        # config_name = get_trt_profile_filename(profile)
        trt_engine_name = os.path.dirname(trt_filename) + "\\v1.5_unet_{}.trt".format(name)

        if os.path.isfile(trt_engine_name):
            print("Skipping engine build for config: {}, engine already exsists. ({})".format(name, trt_engine_name))
            continue

        builder = EngineBuilder(trt_engine_name)
        cache_file = os.path.join(trt_dir, "unet_timing.cache") if EngineBuilder.TRT_REMOTE_CACHE_FILE is None else EngineBuilder.TRT_REMOTE_CACHE_FILE
        print("Using cache file {}".format(cache_file))
        builder.build(onnx_path=onnx_filename, fp16=True, input_profile=profile, timing_cache=cache_file)


    

def generate_trt_engine(trt_filename, onnx_filename, min_bs, opt_bs, max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height, use_fp16, trt_extra_args):

    trt_dir = os.path.dirname(trt_filename)

    cond_dim = get_cond_dim_from_onnx(onnx_filename)
    
    profile = get_unet_trt_profile(cond_dim, min_bs, opt_bs, max_bs, min_token_count, opt_token_count, max_token_count, min_width, opt_width, max_width, min_height, opt_height, max_height)

    
    config_name = get_trt_profile_filename(profile) 

    
    trt_engine_name = trt_dir + "\{}.trt".format(config_name)
    if os.path.isfile(trt_engine_name):
        print("Skipping engine build for config: {}, engine already exsists. ({})".format(config_name, trt_engine_name))
        return ""
    
    print("Building profile {}".format(profile))
    # find a smart way to detect cache file

    builder = EngineBuilder(trt_engine_name)
    
    cache_file = os.path.join(trt_dir, "unet_timing.cache") if EngineBuilder.TRT_REMOTE_CACHE_FILE is None else EngineBuilder.TRT_REMOTE_CACHE_FILE
    print("Using cache file {}".format(cache_file))
    builder.build(onnx_path=onnx_filename, fp16=use_fp16, input_profile=profile, timing_cache=cache_file)

    return ""
   