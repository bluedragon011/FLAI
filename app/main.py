from fastapi import FastAPI
#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel
app = FastAPI()

from utils import detect_hardware_and_load #adjustments to ram and threads number. 


llm = detect_hardware_and_load()


class Query(BaseModel):
    prompt: str
    lang : str = "es"

IDIOMA = {
    "es" : "Responde siempre en español",
    "en" : "Answer always in english"
}
@app.post("/ask") #Post to send the user's prompt
#Function to use when accessed to "http://127.0.0.1:8000/docs"
async def ask_fast(query: Query): #async function. ensures the prompt is an str

    instruccion_idioma = IDIOMA.get(query.lang, IDIOMA["es"])

    prompt = (
        "<|im_start|>system\n"
        "Answer in less than 50 words. ONLY ANSWER WITH KEY WORDS! Before answer, revise that the information provided is correct."
         f"{instruccion_idioma} <|im_end|>\n"
        f"<|im_start|>user\n{query.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    result = llm(
        prompt, 
        max_tokens=80,  #Limit the answer (nº tokens)
        stop=["<|im_end|>", "<|im_start|>", "\nuser"],
        temperature=0.3,
        top_p=0.95, #Cleans the top 5% worst words to use (based on probabilities)
        top_k = 40, #40 1st words used, the rest of them discarted. 
        echo=False
    )


    answer = result["choices"][0]["text"]        

    
    return {"Answer": answer.strip()}
