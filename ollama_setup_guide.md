# Ollama Setup Guide

Complete step-by-step instructions for deploying your fine-tuned Medical SOAP model in Ollama.

## Prerequisites

- Downloaded `llama1b-medical-soap-merged.tar.gz` from Google Colab
- Terminal/Command Prompt access
- ~5GB free disk space

## Step 1: Install Ollama

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download and run the installer from: https://ollama.ai/download/windows

## Step 2: Extract Your Model

Navigate to your Downloads folder:
```bash
cd ~/Downloads
```

Extract the model:
```bash
tar -xzf llama1b-medical-soap-merged.tar.gz
```

You should now have a folder called `llama1b-medical-soap-merged` containing:
- `config.json`
- `generation_config.json`
- `model.safetensors` files
- `tokenizer.json` and related files

## Step 3: Install llama.cpp for Conversion

Clone and build llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

On Windows, you may need to install build tools first.

## Step 4: Convert Model to GGUF Format

Convert the model to GGUF (required format for Ollama):
```bash
python convert.py ~/Downloads/llama1b-medical-soap-merged --outdir ./models --outtype f16
```

This creates: `models/ggml-model-f16.gguf`

## Step 5: Quantize the Model (Recommended)

Quantization reduces model size and improves speed:

```bash
./quantize ./models/ggml-model-f16.gguf ./models/medical-soap-q4.gguf q4_0
```

**Quantization options:**
- `q4_0`: Fastest, smallest (recommended) - ~1GB
- `q5_0`: Balanced - ~1.2GB
- `q8_0`: Highest quality, slower - ~1.8GB

## Step 6: Create Modelfile

In the `llama.cpp` directory, create a file named `Modelfile` (no extension):

```bash
nano Modelfile
```

Paste this content:

```
FROM ./models/medical-soap-q4.gguf

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical assistant. Given a medical dialogue between a patient and healthcare provider, generate a SOAP (Subjective, Objective, Assessment, Plan) summary.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
```

Save and exit:
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter`

## Step 7: Start Ollama Service

Open a **new terminal window** and start Ollama:
```bash
ollama serve
```

Keep this terminal window open. It should show:
```
Ollama server listening on http://127.0.0.1:11434
```

## Step 8: Create the Ollama Model

In your **original terminal** (in the llama.cpp directory), run:
```bash
ollama create medical-soap -f Modelfile
```

Expected output:
```
transferring model data
using existing layer sha256:...
creating new layer sha256:...
writing manifest
success
```

## Step 9: Verify Installation

List your models:
```bash
ollama list
```

You should see `medical-soap` in the list.

## Step 10: Test Your Model

### Interactive Mode

Run the model interactively:
```bash
ollama run medical-soap
```

Test with this prompt:
```
Medical Dialogue:
Doctor: What brings you in today?
Patient: I've been having chest pain for 3 days.
Doctor: Can you describe the pain?
Patient: It's sharp and gets worse when I breathe deeply.
Doctor: Any shortness of breath?
Patient: Yes, especially when walking.
Doctor: Let me examine you. Your heart rate is 95 bpm, blood pressure is 140/85.

Please provide a SOAP summary for this medical dialogue.
```

Exit by typing: `/bye`

### Command Line (Non-Interactive)

```bash
echo "Medical Dialogue: [your dialogue]" | ollama run medical-soap
```

### Python API

```python
import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "medical-soap",
    "prompt": """Medical Dialogue:
Doctor: What brings you in today?
Patient: I've been having headaches.

Please provide a SOAP summary.""",
    "stream": False
}

response = requests.post(url, json=data)
result = response.json()["response"]
print(result)
```

### cURL API

```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "medical-soap",
  "prompt": "Medical Dialogue:\nDoctor: What brings you in?\nPatient: Chest pain.\n\nPlease provide a SOAP summary.",
  "stream": false
}'
```

## Troubleshooting

### "Model not found"
- Run `ollama list` to check if model exists
- Verify you ran `ollama create medical-soap -f Modelfile`
- Check the Modelfile path is correct

### "Connection refused"
- Make sure `ollama serve` is running in another terminal
- Check it's listening on port 11434

### Model generates gibberish
- Try reducing temperature: `PARAMETER temperature 0.3` in Modelfile
- Recreate the model: `ollama rm medical-soap` then `ollama create medical-soap -f Modelfile`

### Out of memory
- Use more aggressive quantization: `q4_0` instead of `q8_0`
- Reduce context length: `PARAMETER num_ctx 1024`

### Conversion errors
- Make sure Python is installed: `python --version`
- Install required packages: `pip install numpy sentencepiece`
- Check model path is correct

## Performance Tuning

### Faster Inference
```
PARAMETER temperature 0.5
PARAMETER num_ctx 1024
PARAMETER num_predict 256
```

### Higher Quality Output
```
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.2
```

## Updating the Model

To update with a new version:
```bash
ollama rm medical-soap
ollama create medical-soap -f Modelfile
```

## Uninstalling

Remove the model:
```bash
ollama rm medical-soap
```

Stop Ollama service:
```bash
# Press Ctrl+C in the terminal running "ollama serve"
```

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Model Quantization Guide](https://github.com/ggerganov/llama.cpp#quantization)

## Support

For issues specific to:
- **This model**: Open an issue on this GitHub repository
- **Ollama**: Check https://github.com/ollama/ollama/issues
- **llama.cpp**: Check https://github.com/ggerganov/llama.cpp/issues
