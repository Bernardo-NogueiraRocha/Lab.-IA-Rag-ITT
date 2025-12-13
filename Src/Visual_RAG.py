import streamlit as st
import pandas as pd
import os
from PIL import Image
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vqa2 import Captioner

# ================= CONFIGURAÇÕES =================
IMAGE_DIR = "Test_images"
CSV_FILE = "Test_images/image_classes.csv"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
# ===============================================

st.set_page_config(page_title="VQA Search Engine", layout="wide")

@st.cache_resource
def load_vqa_and_index():
 
    status_text = st.empty()
    status_text.info("Carregando Modelos e Indexando Imagens... (Isso acontece apenas na primeira execução)")

    df = pd.read_csv(CSV_FILE)
    captioner = Captioner()
    
    documents = []
    
    progress_bar = st.progress(0)
    total_images = len(df)

    # Gerar Legendas
    for idx, row in df.iterrows():
        image_name = row['image_name']
        obj_class = row['object_class']
        image_path = os.path.join(IMAGE_DIR, image_name)

        if os.path.exists(image_path):
            try:
                raw_image = Image.open(image_path).convert('RGB')
                
                # Seleciona perguntas com base na classe (ou fallback para 'car')
                questions_list = captioner.questions.get(obj_class, captioner.questions.get('car'))
                
                generated_text = []
                for q_item in questions_list:
                    question = q_item["q"]
                    inputs = captioner.processor(raw_image, question, return_tensors="pt").to(captioner.device)
                    out = captioner.model.generate(**inputs)
                    answer = captioner.processor.decode(out[0], skip_special_tokens=True)
                    generated_text.append(f"{question} {answer}")

                final_caption = " ".join(generated_text)
                
                # Cria documento com metadados para exibição posterior
                doc = Document(
                    text=final_caption,
                    metadata={
                        "filename": image_name,
                        "class": obj_class,
                        "caption_preview": final_caption
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Erro na imagem {image_name}: {e}")
        
        # Atualiza barra de progresso
        progress_bar.progress((idx + 1) / total_images)

    progress_bar.empty()

    # Criar Índice Vetorial
    embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    
    status_text.success(f"✅ Sistema Pronto! {len(documents)} imagens indexadas.")
    return index

st.title("Recuperação de Imagens com VQA-2")
st.markdown("Digite uma descrição para buscar imagens (ex: *'Show me a red car'* ou *'A big truck'*).")

try:
    index = load_vqa_and_index()
except Exception as e:
    st.error(f"Erro ao carregar o sistema: {e}")
    st.stop()

query = st.text_input("Sua busca:", placeholder="Ex: Show me a picture of a bus")
top_k = st.slider("Número de imagens para retornar:", min_value=1, max_value=10, value=3)

if query:
    st.divider()
    st.subheader(f"Resultados para: '{query}'")
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    if not results:
        st.warning("Nenhuma imagem encontrada com score relevante.")
    else:
        cols = st.columns(len(results))
        
        for i, node in enumerate(results):
            filename = node.metadata["filename"]
            img_class = node.metadata["class"]
            caption = node.metadata["caption_preview"]
            score = node.score
            
            image_path = os.path.join(IMAGE_DIR, filename)
            
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        st.image(img, caption=filename, use_column_width=True)
                    else:
                        st.error(f"Img não encontrada: {filename}")
                
                with col2:
                    st.markdown(f"**Classe Real:** `{img_class}`")
                    st.markdown(f"**Score de Similaridade:** `{score:.3f}`")
                    with st.expander("Ver legenda gerada pelo VQA"):
                        st.write(caption)
                st.divider()