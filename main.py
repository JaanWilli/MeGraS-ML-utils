from flask import Flask, request, send_file
from flask_cors import CORS

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import base64

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    print(content)

    imageurl = content["url"]
    image = Image.open(requests.get(imageurl, stream=True).raw)
    prompt = content["prompt"]
    image = image.convert("RGB")
    inputs = processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.sigmoid(outputs.logits)
    pred = torch.round(pred)
    transform = transforms.ToPILImage()
    output = transform(pred)
    output = output.convert('L')

    output = transforms.functional.resize(output, size=[image.height, image.width])

    output = output.convert("RGBA")
    pixdata = output.load()

    width, height = output.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] != (255, 255, 255, 255):
                pixdata[x, y] = (0, 0, 0, 0)

    buffered = BytesIO()
    output.save(buffered, format="PNG")
    return 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")

app.run()