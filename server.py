# import the necessary packages

import flask
import io
import gc
import os
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS
from keras.applications import imagenet_utils
from keras.models import load_model
from flask_ngrok import run_with_ngrok
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense,Flatten,Input
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D,MaxPooling3D

gc.collect()
graph = tf.get_default_graph()
warnings.simplefilter(action="ignore", category=UserWarning)

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
run_with_ngrok(app)
CORS(app)
model = None
batch_size = 20
input_shape = (299,299)
class_names = ['Stegodon','Stegolophodon ']
num_classes = len(class_names)
data = {"success": False}

def _load_model():
    global model,data
    try:
        if model is None: model = InceptionV3()
        transfer_layer = model.layers[-2]
        transfer_layer.output
        conv_model = Model(inputs=model.input , outputs=transfer_layer.output)
        model = Sequential()
        model.add(conv_model)
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.load_weights("./models/efpcnn-09-07-2562-weights.h5")
    except Exception as e:
        if not "exception" in data:data["exception"] = []
        ex = {"err": str(e)}
        data["exception"].append(ex)

@app.route("/predict", methods=["POST"])
def predict():
    global model,data
    if flask.request.method == "POST":
        if flask.request.files.get("image",''):
            try:
                image = flask.request.files.get('image', '')
                image = Image.open(image)
                img_resized = image.resize(input_shape, Image.LANCZOS)
                img_array = np.expand_dims(np.asarray(img_resized), axis=0)

                with graph.as_default():
                    # preds = model.predict([img_array])
                    preds = model.predict_proba([img_array])
                #calculate score of prediction
                score = -np.sort(-preds[0])*100
                        # score = -np.sort(-preds[0])*100
                data["predictions"] = []

                        # loop over the results and add them to the list of
                        # returned predictions
                # for (imagenetID, label, prob) in preds[0]:
                label = class_names[int(preds.argmax(axis=-1))]
                r = {"label": label, "probability": float(score[0])}
                data["predictions"].append(r)

                data["success"] = True
            except Exception as e:
                if not "exception" in data:data["exception"] = []
                ex = {"err": str(e)}
                data["exception"].append(ex)
                print("ERROR :",str(e))

    return flask.jsonify(data),201

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    _load_model()
    port = int(os.environ.get("PORT", 8000))
    # app.run(host='0.0.0.0', port= port,debug=False)
    app.run()