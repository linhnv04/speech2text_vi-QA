import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# Configure BitsAndBytes for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Model initialization
model_id = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=True,
)

# System prompt
system_prompt = """Bạn là 1 trợ lý thông minh, Hãy trả lời câu hỏi đúng nhất"""

def generate_response(input_text: str, max_length: int = 2000) -> str:
    conversation = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}]
    
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id  
    )
    
    text = tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()
    return text



print(generate_response("xin Chao"))