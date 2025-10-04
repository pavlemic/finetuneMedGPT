"""
Example inference script for the Medical SOAP Note Generator
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path="./llama1b-medical-soap-merged"):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def generate_soap_note(model, tokenizer, dialogue, max_new_tokens=512, temperature=0.7):
    """Generate a SOAP note from a medical dialogue"""
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a medical assistant. Given a medical dialogue between a patient and healthcare provider, generate a SOAP (Subjective, Objective, Assessment, Plan) summary.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Medical Dialogue: {dialogue}
Please provide a SOAP summary for this medical dialogue.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated portion
    soap_note = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return soap_note

def main():
    # Load model
    model, tokenizer = load_model()
    
    # Example dialogues
    examples = [
        {
            "name": "Chest Pain",
            "dialogue": """
Patient: Hi doctor, I've been having chest pain for the past two days.
Doctor: Can you describe the pain? Is it sharp or dull?
Patient: It's a sharp pain that comes and goes, especially when I take deep breaths.
Doctor: Any shortness of breath or sweating?
Patient: Yes, I've been a bit short of breath, but no sweating.
Doctor: Let me examine you. Your heart rate is elevated at 110 bpm, blood pressure is 140/90.
"""
        },
        {
            "name": "Headache",
            "dialogue": """
Doctor: What brings you in today?
Patient: I've been having severe headaches for about a week.
Doctor: Can you describe the headaches?
Patient: They're throbbing, mostly on the right side of my head.
Doctor: Any nausea or sensitivity to light?
Patient: Yes, both. The light really bothers me during the headaches.
Doctor: How often do they occur?
Patient: Almost daily, usually in the afternoon.
Doctor: Your blood pressure is 130/85, which is slightly elevated.
"""
        },
        {
            "name": "Abdominal Pain",
            "dialogue": """
Doctor: Where are you experiencing the most pain?
Patient: All over my belly.
Doctor: How long has this been going on?
Patient: Two to three weeks.
Doctor: Does the pain come and go?
Patient: It does.
Doctor: How would you describe the pain?
Patient: I'd describe it as a gnawing sensation.
Doctor: Is this sensation new?
Patient: I believe so. I don't ever remember feeling this way before.
Doctor: Any past abdominal surgeries?
Patient: None.
"""
        }
    ]
    
    # Generate SOAP notes for each example
    print("\n" + "="*80)
    print("GENERATING SOAP NOTES FROM MEDICAL DIALOGUES")
    print("="*80 + "\n")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}: {example['name']}")
        print(f"{'='*80}\n")
        
        print("DIALOGUE:")
        print("-" * 80)
        print(example['dialogue'].strip())
        print("-" * 80)
        
        print("\nGENERATED SOAP NOTE:")
        print("-" * 80)
        soap_note = generate_soap_note(model, tokenizer, example['dialogue'])
        print(soap_note)
        print("-" * 80)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
