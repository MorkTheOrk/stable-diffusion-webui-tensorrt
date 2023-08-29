import os
import launch

def install():
    if not launch.is_installed("tensorrt"):
        launch.run_pip(f"install nvidia-cudnn-cu11==8.9.4.25", "nvidia-cudnn-cu11")
        launch.run_pip(f"install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.0.post11.dev1", "tensorrt")
        launch.run(['python','-m','pip','uninstall','-y','nvidia-cudnn-cu11'], "removing nvidia-cudnn-cu11")

    if not launch.is_installed("onnx"):
        launch.run_pip(f"install onnx")

    # ONNX parser
    if not launch.is_installed("polygraphy"):
        launch.run_pip(f'install polygraphy --extra-index-url https://pypi.ngc.nvidia.com', "polygraphy")

    # Not yet needed - would allow refitting the onnx model/engine
    # if not launch.is_installed("onnx_graphsurgeon"):
    #     launch.run_pip(f'install onnx_graphsurgeon', "onnx_graphsurgeon")
    

install()
