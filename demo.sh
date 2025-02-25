#!/bin/bash

curl -X 'POST' \
  'http://0.0.0.0:8000/transcribe_and_respond/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@/home/alex/workspace/FPT_OJT/viet-asr/sample/1234_studiovoice.wav'
