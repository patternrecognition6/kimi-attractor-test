#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
import requests
import sys

API_KEY = 'sk-MNnw8SSg1pxzZxAJmQs0MOVJ8vOx95V0MJ7vTvbWvq7CXIrM'
API_URL = 'https://api.moonshot.ai/v1/chat/completions'

print('Testing Moonshot/Kimi API...', flush=True)
print(f'Key: {API_KEY[:10]}...{API_KEY[-5:]}', flush=True)

try:
    r = requests.post(API_URL, headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }, json={
        'model': 'kimi-k2-0905-preview',
        'messages': [{'role': 'user', 'content': 'Say hi in 5 words'}],
        'max_tokens': 50,
    }, timeout=60)
    
    print(f'Status: {r.status_code}', flush=True)
    print(f'Response: {r.text[:1000]}', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)
    sys.exit(1)
