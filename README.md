# RAY
To run the model on the CPU or the GPU with onnx flow,

```
conda activate ryzenai-1.2 //installed with the ryzen1.2 installer
python RAY.py
```

To run on the NPU, first copy this repository to transformers/models/llm_onnx.
Then, 
```
cd <transformers>
conda activate ryzenai-transformers
setup_phx.bat
cd models/llm_onnx
python RAY.py
```

### Note: This assumes that you have the onnx model ready as well as the pytorch quantized model
