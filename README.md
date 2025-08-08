# embedded-ai-model-export
Hands-on project: PyTorch → ONNX → Quantization → Inference (MNIST example)

# PyTorch → ONNX → Quantization Demo

This is a compact hands-on project to explore:
- Training a simple PyTorch CNN on MNIST
- Exporting the model to ONNX
- Quantizing the model to reduce size
- Running inference with ONNX Runtime
- Benchmarking performance of original vs quantized models

## ✅ What I Learned
- How to use `torch.onnx.export()` to convert PyTorch models
- How to apply dynamic quantization using ONNX Runtime
- Inference time vs model size tradeoffs
- Why quantized models benefit embedded hardware more than x86 CPUs

## 🧪 Results

| Model | File Size | Inference Time (avg) |
|-------|-----------|----------------------|
| Original (.onnx) | ~1.3 MB | ~0.023 ms |
| Quantized (uint8) | ~450 KB | ~0.043 ms |

> Note: ONNX Runtime does not optimize ConvInteger ops on CPU. Expect better gains on embedded targets.

## 🧠 Next Steps
- Convert to TensorFlow Lite for embedded deployment
- Deploy to real microcontroller or Raspberry Pi
- Compare with TFLite Micro or Edge Impulse workflows

## 📂 Files
- `pytorch_onnx_quantization_demo.ipynb` — full notebook
- `mnist_model.onnx` — original model
- `mnist_model_quant_uint8.onnx` — quantized model
