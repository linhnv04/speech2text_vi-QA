from transcription import transcript
from gemini import gemini_ans
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",  # Localhost (if you open from localhost in browser)
    "http://127.0.0.1",  # Local IP
    "*",  # Allow all origins (use with caution, more restrictive settings are recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins or specify specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class RequestBody(BaseModel):
    audio: UploadFile
if not os.path.exists("./cache"):
    os.makedirs("./cache")

@app.post("/transcribe_and_respond/")
async def transcribe_and_respond(audio: UploadFile = File(...)):
    try:
        with open("./cache/temp_audio_file", "wb") as buffer:
            buffer.write(await audio.read())

        text = transcript("./cache/temp_audio_file")
        
        response = gemini_ans(text)
        print(response)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

