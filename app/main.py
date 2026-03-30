from fastapi import FastAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel
app = FastAPI()


#------------------PREVIOUS MODEL DEFINITION---------------------------
#Defined outside the prompt-processor func to load the model only once (when server initialized)
#model_pipeline = pipeline(
#    "text-generation", 
#    model="meta-llama/Llama-3.2-1B", 
#    device=-1 
#)
#------------------------------END-------------------------------------

#ACTUAL MODEL DEFINITION (4 BITS) to decrease computational costs (RAM)

class Query(BaseModel): #to receive a JSON
    prompt:str

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

#bnb_config = BitsAndBytesConfig( #not usable unless having an NVIDIA
#    load_in_4bit = True, 
#    bnb_4bit_quant_type="nf4", #normal float 16 it's a type of data optimized for NNs
#    bnb_4bit_compute_dtype=torch.float16 #use 16 bits for operations
#)
#
model = AutoModelForCausalLM.from_pretrained( #Initialize model
    model_id,
    low_cpu_mem_usage=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.post("/ask") #Post to send the user's prompt
#Function to use when accessed to "http://127.0.0.1:8000/ask"
async def ask_fast(query: Query): #async function. ensures the prompt is an str

    full_prompt = f"<|im_start|>system\nAnswer in less than 30 words.<|im_end|>\n<|im_start|>user\n{query.prompt}<|im_end|>\n<|im_start|>assistant\n" #optimized query 4 qwen
    inputs = tokenizer(full_prompt,return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1] #number of exact words of the prompt

    result = model.generate(
        **inputs, 
        max_new_tokens=50,  #Limit the answer (nº tokens)
        num_return_sequences=1,
        #truncation=True #Cuts the answer when the limit is reached to avoid errors.  
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True

    )

    answer = tokenizer.decode(result[0][input_length:], skip_special_tokens=True)
    
    return {"Answer": answer.strip()}