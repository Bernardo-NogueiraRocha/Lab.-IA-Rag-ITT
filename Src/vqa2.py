from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import os
import sys
import time
import csv


class Captioner:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)

        self.questions = {
            "person": [
                {"q": "What is the hair color of the person in the image?", "suffix": "hair"},
                {"q": "What is the clothing color of the person in the image?", "suffix": "clothes"},
                {"q": "What is the physical appearance of the person in the image?", "suffix": "physical appearance"},
            ],
            "car": [
                {"q": "What is the color of the car in the image?", "suffix": "car"},
                {"q": "What is the type of the car that is shown in the image?", "suffix": ""},
            ],
            "truck": [
                {"q": "What is the color of the truck in the image?", "suffix": "truck"},
                {"q": "What is the type of the truck that is shown in the image?", "suffix": "truck"},
            ],
            "motorcycle": [
                {"q": "What is the color of the motorcycle in the image?", "suffix": "motorcycle"},
                {"q": "What is the type of the motorcycle that is shown in the image?", "suffix": ""},
            ],
            "bus": [
                {"q": "What is the color of the bus in the image?", "suffix": "bus"},
                {"q": "What is the type of the bus that is shown in the image?", "suffix": ""},
            ],
        }

    def caption(self, images_to_caption):
        """Generate descriptive tags for each image and record inference times."""
        captions = []
        total_time = 0.0
        multiple = len(images_to_caption) > 1
        csv_file = "results.csv"

        file_exists = os.path.exists(csv_file)

        for img_path, obj_type in images_to_caption.items():
            if not os.path.exists(img_path):
                print(f"Skipping missing file: {img_path}")
                continue

            print(f"\nImage: {img_path}")
            raw_image = Image.open(img_path).convert("RGB")

            q_list = self.questions.get(obj_type.lower())
            if not q_list:
                print(f"Unknown object type: {obj_type}")
                continue

            current_tags = obj_type
            start_time = time.time()

            for item in q_list:
                question = item["q"]
                suffix = item["suffix"]

                inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100)
                answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip().rstrip(".")

                if len(answer.split()) == 1:
                    answer = f"{answer} {suffix}"

                print(f"Q: {question}")
                print(f"A: {answer}")
                current_tags += f", {answer}"

            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"Inference time for {img_path}: {elapsed:.2f} seconds")

            # Save (image, tags, time, class)
            captions.append((img_path, current_tags, round(elapsed, 2), obj_type))

        # Append inference times to CSV
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="|")

            # Add new CSV headers for 'class' and 'tags'
            if not file_exists:
                writer.writerow(["file", "batch", "time", "class", "tags"])

            # Write full information per image
            for path, tags, t, obj_type in captions:
                writer.writerow([path, multiple, f"{t:.2f}", obj_type, tags])

        print("\n=== Summary of Captions ===")
        for path, tags, t, _ in captions:
            print(f"{path} ({t:.2f}s): {tags}")

        print(f"\nTotal inference time: {total_time:.2f} seconds")
        if captions:
            print(f"Average per image: {total_time / len(captions):.2f} seconds")

        return captions


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 blip-vqa.py images/1.jpg,images/2.jpg,... car,car,...")
        sys.exit(1)

    image_paths = sys.argv[1].split(",")
    types_list = sys.argv[2].split(",")

    if len(image_paths) != len(types_list):
        raise ValueError("Number of images and types must match!")

    image_type_map = dict(zip(image_paths, types_list))

    captioner = Captioner()
    captioner.caption(image_type_map)


if __name__ == "__main__":
    main()