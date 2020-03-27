from flask import Flask, render_template, request
from base64 import urlsafe_b64decode
import json
import os

from src.detect import extract_card_info, image_from_bytes

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))


@app.route("/")
def app_demo_camera():
    return render_template("demo_camera.html")


@app.route("/app/demo_camera/submit", methods=["POST", "GET"])
def app_demo_camera_submit():
    img = request.form["img"]

    while len(img) % 4 != 0:
        img += "="

    img = img.split(",")[-1]

    with open("test-dump.txt", "w") as f:
        f.write(img)

    image_data = urlsafe_b64decode(img)

    return json.dumps(extract_card_info(image_from_bytes(image_data)))
