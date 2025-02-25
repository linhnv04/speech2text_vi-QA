from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import sounddevice as sd
import soundfile as sf
from transcription import transcript
from gemini import gemini_ans
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",  
    "http://127.0.0.1", 
    "http://0.0.0.0" 
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL, e.g., "http://127.0.0.1:5500"
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

AUDIO_FILE_PATH = os.path.join("./cache", "temp_audio_file.wav")
if not os.path.exists("./cache"):
    os.makedirs("./cache")

@app.get("/")
async def index():
    return {"message": "Welcome to the FastAPI Audio Service"}

@app.post("/upload_audio")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        with open(AUDIO_FILE_PATH, "wb") as buffer:
            buffer.write(await audio.read())
        return {"message": "Audio file uploaded successfully", "path": AUDIO_FILE_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record_audio")
async def record_audio():
    try:
        samplerate = 16000
        duration = 5  # Duration in seconds
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        sf.write(AUDIO_FILE_PATH, audio_data, samplerate, format='WAV', subtype='PCM_16')
        
        if not os.path.exists(AUDIO_FILE_PATH):
            raise HTTPException(status_code=400, detail="No audio file recorded")
        
        return {"message": "Audio recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording audio: {str(e)}")

@app.get("/transcribe")
async def transcribe():
    try:
        text = transcript(AUDIO_FILE_PATH)
        return {"question": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/respond")
async def respond(request: Request):
    try:
        data = await request.json()
        user_input = data.get("question")
        if user_input == "Không nhận diện được giọng nói.":
            return {"response": "No valid input received."}
        response = gemini_ans(user_input)
        print(response)
        return {"response": response, "question": user_input}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
