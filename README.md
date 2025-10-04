# Medical SOAP Note Generator - Fine-tuned Llama 3.1 1B

A fine-tuned language model for automatically generating SOAP (Subjective, Objective, Assessment, Plan) clinical notes from medical dialogues between doctors and patients.

## Overview

This project fine-tunes the Llama 3.1 1B model using LoRA (Low-Rank Adaptation) on the MTS_Dialogue-Clinical_Note dataset to convert medical conversations into structured clinical documentation.

## Model Details

- **Base Model**: [andrijdavid/Llama3-1B-Base](https://huggingface.co/andrijdavid/Llama3-1B-Base)
- **Dataset**: [har1/MTS_Dialogue-Clinical_Note](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note)
- **Training Method**: LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Framework**: Transformers + PEFT + TRL
- **Training Time**: ~42 minutes on A100 GPU
- **Trainable Parameters**: 3.9M (0.23% of total)

## Features

- Converts medical dialogues into structured SOAP notes
- Memory-efficient training using 4-bit quantization
- Fast inference suitable for clinical workflows
- Can be deployed locally using Ollama

## Requirements

```
transformers
datasets
peft
accelerate
bitsandbytes
trl
torch
huggingface_hub
```

## Installation

```bash
pip install transformers datasets peft accelerate bitsandbytes trl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface_hub
```

## Training

The model was trained with the following configuration:

- **Batch Size**: 1 (with gradient accumulation steps of 4)
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Cosine with warmup
- **Precision**: BFloat16
- **Max Sequence Length**: 1024 tokens

### Training Results

- **Training Loss**: 13.23
- **Validation Loss**: 13.29
- **Dataset Split**: 1,170 training / 131 validation examples

## Usage

### Google Colab Training

1. Open the `train_medical_soap.py` file in Google Colab
2. Ensure you have GPU runtime enabled
3. Login to Hugging Face when prompted
4. Run all cells to train the model
5. Download the fine-tuned model at the end

### Local Inference (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "./llama1b-medical-soap-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)

# Generate SOAP note
dialogue = """
Patient: I've been having chest pain for two days.
Doctor: Can you describe the pain?
Patient: It's sharp and gets worse when I breathe deeply.
Doctor: Any shortness of breath?
Patient: Yes, especially when walking.
"""

prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a medical assistant. Given a medical dialogue between a patient and healthcare provider, generate a SOAP (Subjective, Objective, Assessment, Plan) summary.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Medical Dialogue: {dialogue}
Please provide a SOAP summary for this medical dialogue.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Ollama Deployment

1. Extract the downloaded model:
   ```bash
   tar -xzf llama1b-medical-soap-merged.tar.gz
   ```

2. Convert to GGUF format:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   python convert.py ../llama1b-medical-soap-merged --outdir ./models --outtype f16
   ./quantize ./models/ggml-model-f16.gguf ./models/medical-soap-q4.gguf q4_0
   ```

3. Create Modelfile:
   ```
   FROM ./models/medical-soap-q4.gguf
   
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

4. Create Ollama model:
   ```bash
   ollama create medical-soap -f Modelfile
   ```

5. Run the model:
   ```bash
   ollama run medical-soap
   ```

## Project Structure

```
.
├── train_medical_soap.py       # Main training script
├── requirements.txt             # Python dependencies
├── Modelfile                    # Ollama configuration
├── README.md                    # This file
└── examples/
    └── example_inference.py     # Inference examples
```

## Dataset Information

The MTS_Dialogue-Clinical_Note dataset contains:
- Medical dialogues between doctors and patients
- Corresponding clinical notes
- Various medical sections (CC, HPI, Assessment, Plan)

## Model Architecture

- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.1

- **Quantization**:
  - 4-bit NF4 quantization
  - Double quantization enabled
  - Compute dtype: Float16

## Limitations

- Model is fine-tuned on a relatively small dataset (1,301 examples)
- Best suited for general SOAP note structure, may require validation for specific medical specialties
- Not a substitute for professional medical documentation review
- Should be used as an assistive tool, not as a replacement for clinical judgment

## License

This project uses:
- Llama 3.1 model (subject to Meta's Llama license)
- MTS_Dialogue-Clinical_Note dataset (check dataset license)

## Citation

If you use this model in your research, please cite:

```bibtex
@software{medical_soap_llama_2025,
  author = {Pavle Micic},
  title = {Medical SOAP Note Generator - Fine-tuned Llama 3.1 1B},
  year = {2025},
  url = {https://github.com/yourusername/medical-soap-generator}
}
```

## Acknowledgments

- Meta AI for the Llama 3.1 base model
- Hugging Face for the transformers library
- The creators of the MTS_Dialogue-Clinical_Note dataset

## Disclaimer

This model is for research and educational purposes only. It should not be used for actual clinical documentation without proper validation and oversight by qualified healthcare professionals.
