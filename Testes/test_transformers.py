from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device=0,  # ou "cpu"
    torch_dtype="auto"
)

resposta = pipe(
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "https://exemplo.com/imagem.jpg"},
        {"type": "text", "text": "O que está nesta imagem?"}
      ]
    },
    max_new_tokens=200
)

print(resposta)