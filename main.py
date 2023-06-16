from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import base64

from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

text_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
segment_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)


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
    inputs = text_processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = segment_model(**inputs)

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

@app.route('/caption/<id>', methods=["GET"])
def caption(id):
    print(id)
    caption = predict_step("http://localhost:8080/" + id)
    return caption

@app.route('/caption/<imageid>/c/<segmentid>', methods=["GET"])
def captionSegment(imageid, segmentid):
    print(imageid, segmentid)
    caption = predict_step("http://localhost:8080/" + imageid + "/c/" + segmentid)
    return caption

@app.route('/embedding/<caption>', methods=['GET'])
def embedding(caption):
    print(caption)
    embedding = embedding_model.encode(caption)
    return jsonify(embedding.tolist())


def predict_step(image_path):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    image = Image.open(requests.get(image_path, stream=True).raw)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = caption_model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

app.run()