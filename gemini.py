import json
import httpx
from constants import *
def gemini_ans(info_ques):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    payload = json.dumps({"contents": [{"parts": [{"text": f"{sys_prompt}{info_ques}"}]}]})
    headers = {"Content-Type": "application/json"}

    with httpx.Client(timeout=None) as client:
        response =  client.post(url, headers=headers, data=payload)
    response_data = response.json()
    answer = response_data["candidates"][0]["content"]["parts"][0]["text"]

    return answer



