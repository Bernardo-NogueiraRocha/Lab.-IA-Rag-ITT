import os
from google import genai
from google.genai.errors import APIError
from PIL import Image

def testar_gemini_com_imagem():
    
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não está configurada.")
        print("Execute: export GEMINI_API_KEY='SUA_CHAVE_AQUI'")
        return

    try:
        client = genai.Client()

        modelo = "gemini-2.0-flash"
        prompt = "Descreva esta imagem em detalhes, focando nos objetos e cores principais."
        
        caminho_imagem = "cropped_images/0026a9d9-78c8-41a7-95b8-d49e8a4e0d13_cropped.jpg" 
        
        if not os.path.exists(caminho_imagem):
            print(f"ERRO: O arquivo '{caminho_imagem}' não foi encontrado.")
            print("Por favor, altere a variável 'caminho_imagem' para o caminho correto do seu arquivo.")
            return

        img = Image.open(caminho_imagem)

        contents = [img, prompt]
        
        print(f"PROMPT: {prompt}")

        response = client.models.generate_content(
            model=modelo,
            contents=contents
        )

        print("\n--- 5. Resposta do Gemini recebida! ---")
        print("RESPOSTA:")
        print(response.text)

    except APIError as e:
        print(f"\nERRO DA API: Ocorreu um erro ao chamar a API: {e}")
        print("Verifique se sua chave API é válida e se você tem acesso ao modelo.")
    except Exception as e:
        print(f"\nERRO: Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    testar_gemini_com_imagem()