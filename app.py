from flask import Flask, render_template, request, jsonify
import io
import json
import base64
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

ALLOWED_EXTENSION = {'jpg', 'jpeg'}
model = models.vgg16(weights='IMAGENET1K_V1')
image_index = json.load(open('./imagenet_class_index.json'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


def image_transformation(image_byte):
    img_transformation = transforms.Compose([transforms.Resize(255),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    uploaded_image = Image.open(io.BytesIO(image_byte))
    return img_transformation(uploaded_image).unsqueeze(0)


def prediction(image_byte):
    tensor = image_transformation(image_byte)
    model_output = model.forward(tensor)
    predicted_value = model_output.max(1)[1]
    return image_index[str(predicted_value.item())]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            image_byte = file.read()
            class_id, class_name = prediction(image_byte)
            img_data = base64.b64encode(image_byte).decode('utf-8')
            return render_template('index.html', img_data=img_data, class_id=class_id, class_name=class_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
