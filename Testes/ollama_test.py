import requests
import json
import base64

with open("image.png", "rb") as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode('utf-8')

url = "http://localhost:11434/api/generate"

data = {
    "model": "gemma3:4b",
    "prompt": "Descreva o que há nesta imagem detalhadamente",
    "images": [img_b64],
    "stream": False
}

response = requests.post(url, json=data)

if response.status_code == 200:
    # Se não for streaming
    resp = response.json()
    print(resp.get("response", resp))
else:
    print(f"Erro: {response.status_code} - {response.text}")
