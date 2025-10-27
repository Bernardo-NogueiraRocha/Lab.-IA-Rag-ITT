import os
import sys
import base64
from ollama import chat
import time 


def imagem_para_base64(caminho_imagem: str) -> str:
    """Lê uma imagem e retorna em base64 para enviar ao modelo."""
    with open(caminho_imagem, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def processar_diretorio_com_moondream():
    if len(sys.argv) < 2:
        print("ERRO: O diretório das imagens não foi especificado.")
        print("Uso: python seu_script.py <caminho_do_diretorio>")
        return

    diretorio_imagens = sys.argv[1]

    if not os.path.isdir(diretorio_imagens):
        print(f"ERRO: O caminho '{diretorio_imagens}' não é um diretório válido.")
        return

    modelo = "moondream"  # Modelo leve multimodal da Ollama
    prompt_texto = (
        "Carefully describe the content of this image in detail. Focus on identifying all vehicles, their types, colors, sizes, and any visible distinguishing characteristics such as brand logos, model shapes, or damage. Be objective and concise."
        
    )

    print(f"\nPROMPT GERAL: {prompt_texto}\n")

    for nome_arquivo in os.listdir(diretorio_imagens):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            caminho_completo = os.path.join(diretorio_imagens, nome_arquivo)

            print("-" * 60)
            print(f"Arquivo: {caminho_completo}")

            try:
                img_base64 = imagem_para_base64(caminho_completo)
                antes = time.time()
                resposta = chat(
                    model=modelo,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente especializado em análise de imagens."
                        },
                        {
                            "role": "user",
                            "content": prompt_texto,
                            "images": [img_base64]
                        }
                    ]
                )

                total = time.time() - antes

                print("\nPrevisão:")
                print(resposta["message"]["content"])
                print("Tempo de processamento ", total)
                print("\n")

            except Exception as e:
                print(f"Erro ao processar '{nome_arquivo}': {e}")

if __name__ == "__main__":
    processar_diretorio_com_moondream()