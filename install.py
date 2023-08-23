import os
import launch

def install():
    if not launch.is_installed("tensorrt"):
        launch.run_pip(f"install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.0.post11.dev1", "tensorrt")

    if launch.is_installed("cuda-python"):
        from cuda import cudart
        cudaVersion = cudart.cudaRuntimeGetVersion()[1]
        if cudaVersion != "11080":
            raise RuntimeError("Wrong cuda_python version installed: Got {} expected 11.8.2".format(cudaVersion))
    else:
        launch.run_pip(f'install cuda-python==11.8.2', "cuda-python")

    if not launch.is_installed("polygraphy"):
        launch.run_pip(f'install polygraphy', "polygraphy")

    # Not yet needed - would allow refitting the onnx model/engine
    # if not launch.is_installed("onnx_graphsurgeon"):
    #     launch.run_pip(f'install onnx_graphsurgeon', "onnx_graphsurgeon")
    

install()
