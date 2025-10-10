from google import genai

client = genai.Client()
myfile = client.files.upload(file= "image.png")
print(f"{myfile=}")

result = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        myfile,
        "\n\n",
        "Detail the object of this photo",
    ],
)

print(f"{result.text=}")