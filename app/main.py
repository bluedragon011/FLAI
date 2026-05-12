from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import json
#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pydantic import BaseModel

app = FastAPI()

from utils import detect_hardware_and_load #adjustments to ram and threads number.

#FOR PROCESSING CONTEXT
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

#LOAD: model, embeddings model, database
llm = detect_hardware_and_load()
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = "./chroma_db"

#The class for the Query
class Query(BaseModel):
    prompt: str
    lang : str = "es"
    collection_name: str


#Available languages
IDIOMA = {
    "es" : "Responde siempre en español",
    "en" : "Answer always in english"
}

#function to stream the answer. 
async def dynamic_words(prompt : str):
    for chunk in llm.create_completion(
        prompt,
        max_tokens = 100, 
        stop=["<|im_end|>", "<|im_start|>", "\nuser"], 
        stream=True, #this allows us to send the tokens before all the answer is finished
        temperature = 0.3,
        top_p=0.95, #Cleans the top 5% worst words to use (based on probabilities)
        top_k = 40, #40 1st words used, the rest of them discarted. 
        echo=False
    ):
        token = chunk["choices"][0]["text"] #extract the text from the actual fragment
        yield token #We use this to pause the function until we send the token

#----------------------------------NEW ENDPOINT TO PROCESS USER'S TEXTS/PDFs--------------------------------
@app.post('/upload/{collection_name}')
async def upload_file(collection_name: str, file : UploadFile = File(...)):
    #1st step: save content (temporally)
    with open(file.filename, "wb") as f: 
        shutil.copyfileobj(file.file, f)
    
    #2nd step: Download and process
    loader = PyPDFLoader(file.filename) if file.filename.endswith(".pdf") else TextLoader(file.filename)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) #Splits the text in phrases/words (little context wondow)
    chunks = splitter.split_documents(docs) #Chunks are those fragments. 

    #3rd step: save on chromaDB permanently
    vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_model,
    persist_directory=CHROMA_PATH,
    collection_name=collection_name
)
    os.remove(file.filename)
    return {"status":  "File indexed permanently"}

#----------------------------------ENDPOINT TO PROCESS USER'S QUERY--------------------------------
@app.post("/ask") #Post to send the user's prompt
#Function to use when accessed to "http://127.0.0.1:8000/docs"
async def ask_fast(query: Query): #async function. ensures the prompt is an str

    #New function: add the context saved in the chromaDB: 
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model, collection_name = query.collection_name)
    docs = db.similarity_search(query.prompt, k=3) # 3 best fragments
    context = "\n".join([d.page_content for d in docs])

    instruccion_idioma = IDIOMA.get(query.lang, IDIOMA["es"])

    prompt = (
        "<|im_start|>system\n"
        "Answer in less than 50 words. ONLY ANSWER WITH KEY WORDS! Before answer, revise that the information provided is correct.\n"
         f"{instruccion_idioma}\n"
         f"Context: {context} <|im_end|>\n"
        f"<|im_start|>user\n{query.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    
    return StreamingResponse( #this returns each word instead of all the answer directly. 
        dynamic_words(prompt), #we don't need the answer anymore
        media_type="text/event-stream"
    )
