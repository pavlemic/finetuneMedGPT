# Complete Guide: Fine-tuning Llama 3.1 8B for Medical SOAP Notes

## Step-by-Step Process

### Phase 1: Google Colab Setup and Training

1. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Create a new notebook
   - Change runtime to GPU (Runtime → Change runtime type → GPU → T4/A100)

2. **Get Hugging Face Access**
   - Create account at [huggingface.co](https://huggingface.co)
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
   - Accept the Llama 3.1 license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

3. **Run the Training Code**
   - Copy the Python code from the first artifact into your Colab notebook
   - Run each section step by step
   - When prompted, enter your Hugging Face token
   - Training will take 2-4 hours depending on GPU

4. **Monitor Training**
   - Watch the loss decrease over time
   - Training logs will show progress every 10 steps
   - Model will be saved every 500 steps

5. **Download Your Model**
   - The script will automatically compress and download your fine-tuned model
   - Save the `llama-medical-soap-merged.tar.gz` file to your local machine

### Phase 2: Ollama Integration

#### Option 1: Using Ollama with HuggingFace Models (Recommended)

1. **Install Ollama**
   ```bash
   # On macOS
   brew install ollama
   
   # On Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows - download from ollama.ai
   ```

2. **Extract Your Model**
   ```bash
   tar -xzf llama-medical-soap-merged.tar.gz
   ```

3. **Create Ollama Modelfile**
   Create a file called `Modelfile`:
   ```
   FROM ./llama-medical-soap-merged
   
   TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

   You are a medical assistant. Given a medical dialogue between a patient and healthcare provider, generate a SOAP (Subjective, Objective, Assessment, Plan) summary.

   <|eot_id|><|start_header_id|>user<|end_header_id|>

   {{ .Prompt }}

   <|eot_id|><|start_header_id|>assistant<|end_header_id|>

   """
   
   PARAMETER stop "<|eot_id|>"
   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   ```

4. **Create the Ollama Model**
   ```bash
   ollama create medical-soap -f Modelfile
   ```

5. **Test Your Model**
   ```bash
   ollama run medical-soap "Patient: I've been having headaches for a week. Doctor: How severe are they? Patient: Pretty bad, maybe 7/10. Doctor: Any nausea? Patient: Yes, especially in the mornings."
   ```

#### Option 2: Direct GGUF Conversion (Advanced)

1. **Install llama.cpp**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```

2. **Convert to GGUF**
   ```bash
   python convert.py /path/to/llama-medical-soap-merged --outdir ./models --outtype f16
   ```

3. **Quantize (Optional)**
   ```bash
   ./quantize ./models/ggml-model-f16.gguf ./models/ggml-model-q4_0.gguf q4_0
   ```

4. **Import to Ollama**
   ```bash
   ollama create medical-soap -f Modelfile-gguf
   ```

### Phase 3: Usage Examples

#### Testing Your Medical SOAP Model

```bash
# Example 1: Simple consultation
ollama run medical-soap "
Medical Dialogue:
Patient: Doctor, I've been feeling very tired lately and have frequent headaches.
Doctor: How long have you been experiencing these symptoms?
Patient: About two weeks now. The headaches are worse in the morning.
Doctor: Any changes in your vision or nausea?
Patient: No vision changes, but I do feel nauseous sometimes.
Doctor: Let me check your blood pressure. *measures* It's 150/95, which is elevated.

Please provide a SOAP summary for this medical dialogue.
"
```

Expected output format:
```
**SOAP Summary:**

**Subjective:**
- Patient reports fatigue and frequent headaches for approximately 2 weeks
- Headaches more severe in the morning
- Occasional nausea
- Denies vision changes

**Objective:**
- Blood pressure: 150/95 mmHg (elevated)

**Assessment:**
- Hypertension with associated headaches and fatigue
- Morning headaches possibly related to elevated blood pressure

**Plan:**
- Further evaluation of hypertension
- Consider antihypertensive medication
- Follow-up appointment recommended
- Monitor blood pressure at home
```

### Troubleshooting Common Issues

1. **Out of Memory in Colab**
   - Reduce batch size to 1
   - Increase gradient accumulation steps
   - Use smaller LoRA rank (r=8 instead of r=16)

2. **Ollama Model Not Loading**
   - Check that the model files are in the correct format
   - Ensure sufficient RAM (8GB+ recommended for 8B model)
   - Try quantized version for lower memory usage

3. **Poor Model Performance**
   - Train for more epochs
   - Adjust learning rate
   - Use more training data
   - Fine-tune LoRA parameters

### Model Performance Tips

1. **For Better Medical Accuracy**
   - Train for more epochs (5-10)
   - Use medical-specific evaluation metrics
   - Add medical terminology to the tokenizer if needed

2. **For Faster Inference**
   - Use quantized versions (Q4_0 or Q8_0)
   - Consider smaller models for production use
   - Optimize hardware (GPU inference when possible)

3. **For Better SOAP Format**
   - Add more structured examples to training
   - Use consistent formatting in prompts
   - Consider post-processing to enforce format

### Production Deployment

1. **API Server Setup**
   ```bash
   # Start Ollama server
   ollama serve
   
   # Use REST API
   curl -X POST http://localhost:11434/api/generate -d '{
     "model": "medical-soap",
     "prompt": "Your medical dialogue here...",
     "stream": false
   }'
   ```

2. **Integration Examples**
   - Python client using `ollama` package
   - REST API integration for web applications
   - Batch processing for multiple dialogues

This complete workflow gives you a production-ready medical SOAP note generation model fine-tuned specifically on medical dialogue data.
