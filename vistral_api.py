from vistral7b import generate_response
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app = FastAPI()

origins = [
    "http://localhost",  
    "http://127.0.0.1", 
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)




class TextRequest(BaseModel):
    text: str


@app.post("/transcribe")
async def transcribe_text(request: TextRequest):
    vistral_resp = generate_response(request.text)
    return  JSONResponse(content={"response": vistral_resp})

