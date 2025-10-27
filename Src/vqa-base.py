from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import os
import time

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

directory = sys.argv[1]
question = sys.argv[2]

print("Questão:", question)

for file in os.listdir(directory):
    if file.endswith('.jpg'):
        print("Arquivo:", directory + file)
        
        raw_image = Image.open(directory+ file).convert('RGB')
        start = time.time()
        inputs = processor(images=raw_image, text=question, return_tensors="pt").to("cuda")
        
        out = model.generate(**inputs, max_new_tokens=50)
        end = time.time()
        print("Previsão:", processor.decode(out[0], skip_special_tokens=True))
        print("Tempo de processamento:", end - start)