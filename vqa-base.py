from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

import os
for file in os.listdir('cropped_images'):
    if file.endswith('.jpg'):
        print("Arquivo:",file)
        raw_image = Image.open('cropped_images/'+ file).convert('RGB')
        #comente essas 2 linhas e tire o comentário para testar sem questão
        question = "How many cars in the picture?"
        inputs = processor(raw_image, question,return_tensors="pt").to("cuda")
        #inputs = processor(raw_image,return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        print("Previsão:",processor.decode(out[0], skip_special_tokens=True))