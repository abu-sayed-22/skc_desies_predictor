"""
ONly test purpose

"""



import os
from flask import Flask, request, render_template, redirect, url_for, make_response
import base64
# PILfor Image
from PIL import Image
from io import BytesIO
import json
# loiading the learner
from fastcore.all import *
from fastai.learner import load_learner

app = Flask(__name__)


@app.get('/')
def home():
    return {
        "wlc": "something is gonna update soon"
    }


# defining the predictroue
@app.post('/predict')
def predict():
    """
    things that we should recive
    {
        "image":"base64code",
        "apikey":"NOthing"
    }

    """
    data = json.loads(request.data)
    base64_image_data = data["image"]
    # at first convert the base64 image into pil formate image
    try:
        binary_data = base64.b64decode(base64_image_data)
        image = Image.open(BytesIO(binary_data))
        with open("hello.jpg", 'wb') as img:
            img.write(binary_data)
        # image.show()
    except (base64.binascii.Error, OSError, Exception) as e:
        return make_response({"error": "500", "message": "Conversion is not poossible"}, 500)

    # now fit to the model and return the result to the user
    learner = load_learner("./export.pkl")
    # result =learner.predict(image)
    pred_class, pred_idx, outputs = learner.predict('hello.jpg')
    os.unlink('./hello.jpg')

    return {
        "output": {
            "pred_class": pred_class
        }
    }


# @app.route()
# def error_route():
#     return{
#         "error":"Generalize",
#     }
# finally run the app
if __name__ == "__main__":
    app.run(debug=False)