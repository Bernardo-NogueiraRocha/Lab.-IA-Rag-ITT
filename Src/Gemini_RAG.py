import os
import re
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_vqa_results_to_documents(filepath: str):
    """Parse VQA results text file and return a list of Documents."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\nImage:\s*", content)
    docs = []

    for section in sections:
        if not section.strip():
            continue

        match = re.match(r"([^\n]+)", section)
        if not match:
            continue
        image_path = match.group(1).strip()

        qa_pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)\n", section, re.DOTALL)
        caption_parts = [f"{q.strip()} {a.strip()}" for q, a in qa_pairs]
        caption_text = " ".join(caption_parts)

        if caption_text:
            doc = Document(text=caption_text, metadata={"image": image_path})
            docs.append(doc)

    print(f"Loaded {len(docs)} image documents from {filepath}")
    return docs

def main():
    docs = load_vqa_results_to_documents("results/Test_images/test2/results_vqa2.txt")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")
    llm = GoogleGenAI(api_key=gemini_api_key, model="gemini-2.0-flash")

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    query = "white delivery truck"
    print(f"\nQuery: {query}\n{'-'*40}")
    response = query_engine.query(query)

    for node in response.source_nodes:
        print(node.metadata["image"], "→", node.text)


if __name__ == "__main__":
    main()
