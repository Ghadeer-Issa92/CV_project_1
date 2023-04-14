from flask import Flask , jsonify, request 
import io 
import json
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms 


app =Flask(__name__)
ALLOWED_EXTENSION = {'jpg','jpeg'}
model = models.vgg16(weights='imagenet')#models.vgg16(pretrained =True)
image_index=json.load(open('./imagenet_class_index.json'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION


def image_transformation(image_byte):
    img_transformation=transforms.Compose([transforms.Resize(255),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    uploaded_image=Image.open(io.BytesIO(image_byte))
    return img_transformation(uploaded_image).unsqueeze(0)

def prediction(image_byte):
    tensor=image_transformation(image_byte)
    model_output = model.forward(tensor)
    predicted_value=model_output.max(1)[1]
    return image_index[str(predicted_value.item())]


@app.route('/', methods=['GET','POST'] )
def index():
    return ("Welcome")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            image_byte = file.read()
            class_id , calss_name = prediction(image_byte)
            return jsonify({'class id':class_id,'class name':calss_name})
    return "no prediction. Sorry!!"
            


if __name__ ==  '__main__':
    app.run()
