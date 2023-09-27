import os
import launch
from logging import info

def install():
    if not launch.is_installed("tensorrt"):
        info("TensorRT is not installed! Installing...")
        launch.run_pip(f"install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4", "tensorrt")

    if not launch.is_installed("onnx"):
        info("ONNX is not installed! Installing...")
        launch.run_pip(f"install onnx")

    # Polygraphy	
    if not launch.is_installed("polygraphy"):
        info("Polygraphy is not installed! Installing...")
        launch.run_pip(f'install polygraphy --extra-index-url https://pypi.ngc.nvidia.com', "polygraphy")

    # ONNX GS
    if not launch.is_installed("onnx-graphsurgeon"):
        info("GS is not installed! Installing...")
        launch.run_pip(f'install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com', "onnx-graphsurgeon")

    # TODO optional 
    # onnxruntime==1.15
install()
