import os
import sys
import base64
from ollama import chat
from PIL import Image

def imagem_para_base64(caminho_imagem: str) -> str:
    """Lê uma imagem e retorna em base64 para enviar ao modelo."""
    with open(caminho_imagem, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def processar_diretorio_com_gemma():
    if len(sys.argv) < 2:
        print("ERRO: O diretório das imagens não foi especificado.")
        print("Uso: python seu_script.py <caminho_do_diretorio>")
        return

    diretorio_imagens = sys.argv[1]

    if not os.path.isdir(diretorio_imagens):
        print(f"ERRO: O caminho '{diretorio_imagens}' não é um diretório válido.")
        return

    modelo = "gemma3:4b"
    prompt_texto = "Descreva esta imagem em detalhes, de forma sucinta, focando nos objetos e cores principais."
    print(f"PROMPT GERAL: {prompt_texto}\n")

    for nome_arquivo in os.listdir(diretorio_imagens):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            caminho_completo = os.path.join(diretorio_imagens, nome_arquivo)

            print("-" * 50)
            print(f"PROCESSANDO ARQUIVO: {nome_arquivo}")

            try:
                img_base64 = imagem_para_base64(caminho_completo)

                resposta = chat(
                    model=modelo,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente de visão computacional."
                        },
                        {
                            "role": "user",
                            "content": prompt_texto,
                            "images": [img_base64]
                        }
                    ]
                )

                print("RESPOSTA DO GEMMA:")
                print(resposta["message"]["content"])

            except FileNotFoundError:
                print(f"AVISO: O arquivo '{nome_arquivo}' não foi encontrado. Pulando.")
            except Exception as e:
                print(f"AVISO: Não foi possível processar '{nome_arquivo}'. Erro: {e}")

if __name__ == "__main__":
    processar_diretorio_com_gemma()
