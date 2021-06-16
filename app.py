from flask import Flask, request, jsonify, render_template

from predict import  ResNet9, get_default_device, to_device, loadcheckPoint, predict_image
from torchvision.datasets import ImageFolder  # for working with classes and images
import torchvision.transforms as transforms   # for transforming images into tensors 
from PIL import Image
import os
import io



app = Flask(__name__)




def get_prediction(tensor):
    '''
    geting the prediction from the device
    '''
    device = get_default_device()
    model = to_device(ResNet9(3, 38), device) 
    nn_filename = 'checkpoint.pth'
    loaded_model = loadcheckPoint(nn_filename, model)
    # loadcheckPoint(,model)
    predicted=predict_image(tensor,model)
    return predicted




def transform_image(image_location):

    '''
    converting the image into the tensor
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    image_bytes = Image.open(image_location)
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.route('/')
def home():
    return render_template('index.html')

UPLOAD_FOLDER = "IMAGE_FOLDER/"
@app.route('/',methods=["POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            # image=transform_image(image_location)
            trans1 = transforms.ToTensor()
            im = Image.open(image_location)

            disease= get_prediction(trans1(im))
            return render_template("index.html",prediction = disease)
    return render_template("index.html", prediciton = 1)

if __name__ == "__main__":
    app.run(debug=True)




