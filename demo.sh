#!/bin/bash

curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe_and_respond/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@/home/alex/workspace/FPT_OJT/viet-asr/sample/1IJPK91LV_48BTAM.mp3'
