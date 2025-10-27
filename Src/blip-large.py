from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


directory = sys.argv[1]

import os
import time
for file in os.listdir(directory):
     if file.endswith('.jpg'):
        print("Arquivo:",directory + file)
        raw_image = Image.open('Test_images/'+ file).convert('RGB')
        question = "I see a"
        start = time.time()
        inputs = processor(raw_image, question,return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50)
        end = time.time()
        print("Previsão:",processor.decode(out[0], skip_special_tokens=True))
        print("Tempo de processamento:",end - start)