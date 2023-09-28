import launch

def install():
    if not launch.is_installed("tensorrt"):
        print("TensorRT is not installed! Installing...")
        launch.run_pip("install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-deps --no-cache-dir --force-reinstall", "tensorrt", live=True)

    # Polygraphy	
    if not launch.is_installed("polygraphy"):
        print("Polygraphy is not installed! Installing...")
        launch.run_pip("install polygraphy --extra-index-url https://pypi.ngc.nvidia.com", "polygraphy", live=True)

    # ONNX GS
    if not launch.is_installed("onnx-graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install protobuf==3.20.2", "protobuf", live=True)
        launch.run_pip('install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com', "onnx-graphsurgeon", live=True)

    # TODO optional 
    # onnxruntime==1.15
install()