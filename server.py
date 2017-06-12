#!/usr/bin/env python3

"""Hololens backend server implemented with Flask"""

import pickle
import hashlib
import datetime
import json
import configuration
import logger
import classifierFace
from flask import Flask
from flask import request
import cv2

# Instantiate and configure Flask server

server = Flask("Hololens backend server")
server.config['DEBUG'] = configuration.FLASK_DEBUG

# Instantiate face classifier

logger.log_event("Server: begin loading face classifier")
classifier_face = classifierFace.ClassifierFace()
logger.log_event("Server: finished loading face classifier")

# API endpoints

# Template endpoint, e.g. admin, reload models

@server.route("/api/admin/reload", methods=["POST"])
def route_admin_reload():

    """Template endpoint"""

    if request.method == "POST":

        json_dict = request.get_json()
        command = int(json_dict['command'])

        # Code

        # TODO Implement functionality

        # Response

        response = {"message": "ok"}
        return json.dumps(response)

    else:
        response = {"message": "bad request"}
        return json.dumps(response)


# Detect faces

@server.route("/api/detect/faces", methods=["POST"])
def route_detect_faces():

    """Detect faces"""

    logger.log_event("Server: received face detection request")

    if request.method == "POST":

        # Receive data and deserialize to image matrix (marked below with horizontal rules)
        # NOTE: THIS DESERIALIZATION NEEDS TO MATCH THE CLIENT (SENDER'S) SERIALIZATION

        # ---------------------------------------------------------------------

        image_matrix_pickled = request.get_data()
        # Unpickle from byte string
        image_matrix = pickle.loads(image_matrix_pickled)

        # ---------------------------------------------------------------------

        # Write image to disk (optionally)
 
        if configuration.DEBUG_SERVER_WRITE_IMAGE:
            timestamp = '{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
            image_filename = timestamp + ".jpg"
            cv2.imwrite(configuration.DEBUG_SERVER_WRITE_IMAGE_DIR + "/" + image_filename, image_matrix)
            logger.log_event("Server: image written to disk at " + configuration.DEBUG_SERVER_WRITE_IMAGE_DIR + "/" + image_filename)

        # Compute MD5 hash and log it

        hash_md5 = hashlib.md5()
        hash_md5.update(image_matrix)
        hash_image_matrix = hash_md5.hexdigest()
        logger.log_event("Server: received image matrix MD5 hash is: " + hash_image_matrix)

        # Detect faces and classify faces

        bounding_boxes, predictions = classifier_face.classify(image_matrix)
        logger.log_event("Server: finished face detection on image")

        # Build response and send it

        response = {"bounding_boxes": bounding_boxes, "predictions": predictions}
        return json.dumps(response)

    else:
        logger.log_error("Server: received a bad requesst")
        response = {"message": "bad request"}
        return json.dumps(response)


def main():

    """Run Flask server application"""

    logger.log_event("Server: starting Flask server")
    server.run(host=configuration.FLASK_HOST, port=configuration.FLASK_PORT)


if __name__ == "__main__":

    main()
