# embedded-ai-model-export
Hands-on project: PyTorch → ONNX → Quantization → Inference (MNIST example)

# PyTorch → ONNX → Quantization Demo (MNIST)

This is a hands-on mini-project to demonstrate:
- Training a simple CNN using PyTorch on the MNIST dataset
- Exporting the model to ONNX format
- Quantizing the ONNX model to reduce its size
- Running inference using ONNX Runtime
- Benchmarking performance of original vs quantized models

---

## ✅ What I Learned

- How to convert a PyTorch model to ONNX
- How to apply dynamic quantization with ONNX Runtime
- How quantization affects model size and inference time
- Practical insights into edge/embedded AI deployment pipelines

---

## 🧪 Results

### Prediction:
Expected inference for: (digit 4)
![Digit Prediction](./digit_prediction_example.png)

Outcome:
output: ONNX Prediction: 4 | Actual Label: 4

### Execution time comparison:

| Model                 | Size     | Inference Time (CPU) |
|----------------------|----------|-----------------------|
| `mnist_model.onnx`   | ~1.3 MB  | ~0.023 ms             |
| `mnist_model_quant_uint8.onnx` | ~450 KB | ~0.043 ms  |

> ⚠️ ONNX Runtime on x86 doesn’t accelerate ConvInteger ops. Better gains expected on ARM/embedded targets.

---

## 📂 Project Structure

embedded-ai-model-export/
├── models/
│ ├── mnist_model.onnx
│ └── mnist_model_quant_uint8.onnx
├── pytorch_onnx_quantization_demo.ipynb
├── README.md

---

## 🚀 Next Steps

- Try converting to TFLite and run on microcontrollers
- Test ONNX inference on Raspberry Pi or ARM boards
- Explore static quantization or QAT

---

## 🧠 Author

**Shravan Suryanarayana**  
System Software Architect | Embedded AI Explorer  
[LinkedIn](https://linkedin.com/in/shravansurya)
