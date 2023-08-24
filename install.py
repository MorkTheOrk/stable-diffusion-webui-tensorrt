import os
import launch

def install():
    if not launch.is_installed("tensorrt"):
        launch.run_pip(f"install --pre --index-url https://pypi.nvidia.com tensorrt==9.0.0.post11.dev1", "tensorrt")

    # ONNX parser
    if not launch.is_installed("polygraphy"):
        launch.run_pip(f'install polygraphy', "polygraphy")

    # Not yet needed - would allow refitting the onnx model/engine
    # if not launch.is_installed("onnx_graphsurgeon"):
    #     launch.run_pip(f'install onnx_graphsurgeon', "onnx_graphsurgeon")
    

install()
