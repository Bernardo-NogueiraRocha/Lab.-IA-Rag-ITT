import os
from google import genai
from google.genai.errors import APIError
from PIL import Image

def testar_gemini_com_imagem():
    """
    Carrega uma imagem local e usa o modelo Gemini para gerar uma descrição.
    A chave API é lida da variável de ambiente GEMINI_API_KEY.
    """
    
    # 1. Tenta obter a chave API (lida automaticamente pela biblioteca)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não está configurada.")
        print("Execute: export GEMINI_API_KEY='SUA_CHAVE_AQUI'")
        return

    print("--- 1. Chave API encontrada. ---")

    try:
        # 2. Inicializa o cliente GenAI
        client = genai.Client()
        print("--- 2. Cliente Gemini inicializado. ---")

        # 3. Definições
        modelo = "gemini-2.0-flash"
        prompt = "Descreva esta imagem em detalhes, focando nos objetos e cores principais."
        
        # --- ATENÇÃO: SUBSTITUA PELO CAMINHO DA SUA IMAGEM ---
        caminho_imagem = "cropped_images/0026a9d9-78c8-41a7-95b8-d49e8a4e0d13_cropped.jpg" 
        
        # 4. Verifica e Carrega a Imagem
        if not os.path.exists(caminho_imagem):
            print(f"ERRO: O arquivo '{caminho_imagem}' não foi encontrado.")
            print("Por favor, altere a variável 'caminho_imagem' para o caminho correto do seu arquivo.")
            return

        print(f"\n--- 3. Carregando imagem de: {caminho_imagem} ---")
        
        # Carrega a imagem usando PIL
        img = Image.open(caminho_imagem)

        # Prepara o conteúdo (a lista deve conter primeiro a imagem e depois o texto)
        contents = [img, prompt]
        
        print(f"\n--- 4. Enviando Imagem e Prompt para o modelo {modelo}... ---")
        print(f"PROMPT: {prompt}")

        # 5. Chama a API
        response = client.models.generate_content(
            model=modelo,
            contents=contents
        )

        # 6. Exibe o resultado
        print("\n--- 5. Resposta do Gemini recebida! ---")
        print("RESPOSTA:")
        print(response.text)
        print("\n--- Teste concluído com sucesso! ---")

    except APIError as e:
        print(f"\nERRO DA API: Ocorreu um erro ao chamar a API: {e}")
        print("Verifique se sua chave API é válida e se você tem acesso ao modelo.")
    except Exception as e:
        print(f"\nERRO: Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    testar_gemini_com_imagem()