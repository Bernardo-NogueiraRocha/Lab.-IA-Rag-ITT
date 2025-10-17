import os
import sys
from google import genai
from google.genai.errors import APIError
from PIL import Image

def processar_diretorio_com_gemini():
    
    if len(sys.argv) < 2:
        print("ERRO: O diretório das imagens não foi especificado.")
        print("Uso: python seu_script.py <caminho_do_diretorio>")
        return

    diretorio_imagens = sys.argv[1]

    if not os.path.isdir(diretorio_imagens):
        print(f"ERRO: O caminho '{diretorio_imagens}' não é um diretório válido.")
        return
        
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não está configurada.")
        print("Execute: export GEMINI_API_KEY='SUA_CHAVE_AQUI'")
        return

    try:
        client = genai.Client()
        modelo = "gemini-2.0-flash"
        
        prompt_texto = "Descreva esta imagem em detalhes, de forma sucinta, focando nos objetos e cores principais."
        print(f"PROMPT GERAL: {prompt_texto}\n")
        
        for nome_arquivo in os.listdir(diretorio_imagens):
            if nome_arquivo.lower().endswith(('.jpg', '.jpeg')):
                
                caminho_completo = os.path.join(diretorio_imagens, nome_arquivo)
                
                print("-" * 50)
                print(f"PROCESSANDO ARQUIVO: {nome_arquivo}")

                try:
                    img = Image.open(caminho_completo)
                    contents = [img, prompt_texto]
                    
                    response = client.models.generate_content(
                        model=modelo,
                        contents=contents
                    )

                    print("RESPOSTA DO GEMINI:")
                    print(response.text)

                except FileNotFoundError:
                    print(f"AVISO: O arquivo '{nome_arquivo}' não foi encontrado. Pulando.")
                except Exception as e:
                    print(f"AVISO: Não foi possível processar '{nome_arquivo}'. Erro: {e}")
                    
    except APIError as e:
        print(f"\nERRO DA API: Ocorreu um erro ao chamar a API: {e}")
        print("Verifique se sua chave API é válida e se você tem acesso ao modelo.")
    except Exception as e:
        print(f"\nERRO: Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    processar_diretorio_com_gemini()