import launch

def install():
    if not launch.is_installed("tensorrt"):
        print("TensorRT is not installed! Installing...")
        launch.run_pip("install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-deps --no-cache-dir", "tensorrt", True)

    if not launch.is_installed("onnx"):
        print("ONNX is not installed! Installing...")
        launch.run_pip("install onnx")

    # Polygraphy	
    if not launch.is_installed("polygraphy"):
        print("Polygraphy is not installed! Installing...")
        launch.run_pip("install polygraphy --extra-index-url https://pypi.ngc.nvidia.com", "polygraphy", True)

    # ONNX GS
    if not launch.is_installed("onnx-graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com", "onnx-graphsurgeon", True)
        
install()