import pandas as pd
import os
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vqa2 import Captioner
from PIL import Image

# =================CONFIGURAÇÕES=================
CSV_FILE = "Test_images/image_classes.csv"
IMAGE_DIR = "Test_images"
TOP_K = 5  # Quantas imagens recuperar por busca
# ===============================================

def generate_captions_from_csv(df, captioner):

    documents = []
    print(f"--- Iniciando geração de captions para {len(df)} imagens ---")

    for index, row in df.iterrows():
        image_name = row['image_name']
        obj_class = row['object_class']
        image_path = os.path.join(IMAGE_DIR, image_name)

        # Verificação básica de existência
        if not os.path.exists(image_path):
            print(f"Aviso: Imagem {image_path} não encontrada. Pulando.")
            continue

        try:
            # Carrega imagem
            raw_image = Image.open(image_path).convert('RGB')
            
            # Seleciona perguntas baseadas na classe do CSV
            if obj_class in captioner.questions:
                questions_list = captioner.questions[obj_class]
            else:
                # Fallback se a classe não estiver no dicionário (ex: usa perguntas de 'car')
                questions_list = captioner.questions.get('car', [])

            generated_text = []
            
            # Loop de inferência (QA)
            for q_item in questions_list:
                question = q_item["q"]
                inputs = captioner.processor(raw_image, question, return_tensors="pt").to(captioner.device)
                out = captioner.model.generate(**inputs)
                answer = captioner.processor.decode(out[0], skip_special_tokens=True)
                
                # Monta o texto final: "Pergunta? Resposta."
                generated_text.append(f"{question} {answer}")

            final_caption = " ".join(generated_text)
            
            # Cria documento para o LlamaIndex
            # Metadata é crucial para a avaliação (saber qual imagem é qual)
            doc = Document(
                text=final_caption, 
                metadata={"filename": image_name, "ground_truth_class": obj_class}
            )
            documents.append(doc)
            print(f"[OK] {image_name} ({obj_class}): {final_caption[:50]}...")

        except Exception as e:
            print(f"[ERRO] Falha ao processar {image_name}: {e}")

    return documents

def evaluate_retrieval(index, unique_classes, df_ground_truth):
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    
    print("\n=== RELATÓRIO DE AVALIAÇÃO DE RECUPERAÇÃO ===")
    
    overall_precision = []

    for cls in unique_classes:
        query = f"Show me a picture of a {cls}"
        print(f"\nQuery: '{query}'")
        
        results = retriever.retrieve(query)
        
        hits = 0
        for node in results:
            retrieved_filename = node.metadata["filename"]
            retrieved_class = node.metadata["ground_truth_class"]
            score = node.score
            
            # Verifica se a classe da imagem recuperada bate com a query
            is_match = (retrieved_class == cls)
            match_str = "CORRETO" if is_match else "ERRADO"
            
            if is_match:
                hits += 1
                
            print(f"  -> Recuperado: {retrieved_filename} | Classe Real: {retrieved_class} | Score: {score:.3f} | [{match_str}]")
        
        precision = hits / len(results) if results else 0
        overall_precision.append(precision)
        print(f"  Precision@{TOP_K} para '{cls}': {precision:.2f}")

    avg_precision = sum(overall_precision) / len(overall_precision) if overall_precision else 0
    print(f"\n>>> Média de Precision@{TOP_K} Geral: {avg_precision:.2f}")

def main():
    df = pd.read_csv(CSV_FILE)
    unique_classes = df['object_class'].unique()
    print(f"Classes encontradas: {unique_classes}")

    captioner = Captioner() 

    docs = generate_captions_from_csv(df, captioner)

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    evaluate_retrieval(index, unique_classes, df)

if __name__ == "__main__":
    main()