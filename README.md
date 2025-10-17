# Projeto RAG-ITT

RAG : Retrieval Augmented Generation  

ITT: Image-to-Text

Para gerar o environment, usei:  
``conda env create -f environment.yml``  
``conda activate blip-env``  
``python -c "import torch; print(torch.cuda.is_available())"`` (Para verificar CUDA)  
e ``python blip-base.py > results_blip_base.txt``  

Para deletar um modelo da cache é preciso:  
``pip install -U "huggingface_hub[cli]"``

Seguido de:
``hf cache delete``

Para executar um teste, basta usar:  
``python Src/(blip-large, blip-base, test_gemini).py (diretório com imagens do teste)``  
Para usar o VQA-base, passar a questão por argumento (não esquecer das aspas ""), junto do diretório de imagens:  
``python Src/vqa-base.py (diretório) "(pergunta)"``  

Se quiser redirecionar, usar o ``> arquivo.txt``
