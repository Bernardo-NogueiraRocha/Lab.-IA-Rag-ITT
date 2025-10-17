from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

directory = sys.argv[1]
question = sys.argv[2]

import os
for file in os.listdir(directory):
     if file.endswith('.jpg'):
        print("Arquivo:",directory + file)
        raw_image = Image.open('Test_images/'+ file).convert('RGB')
        inputs = processor(raw_image, question,return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        print("Previsão:",processor.decode(out[0], skip_special_tokens=True))
