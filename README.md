# Projeto RAG-ITT

RAG : Retrieval Augmented Generation  

ITT: Image-to-Text

Para gerar o environment, usei:  
``conda env create -f environment.yml``  
``conda activate blip-env``  
``python -c "import torch; print(torch.cuda.is_available())"`` (Para verificar CUDA)  
e ``python blip-base.py > results_blip_base.txt``  
