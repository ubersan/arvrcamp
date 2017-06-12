#!/usr/bin/env python3

"""Configuration"""

# Flask

FLASK_DEBUG = False
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Model directories

HAARCASCADE_MODEL_DIR = "./model/haarcascade"
MTCNN_MODEL_DIR = "./model/mtcnn"
FACEEMBEDDING_MODEL_DIR = "./model/facenet"
SVM_MODEL_DIR = "./model/svm"

# Debug

DEBUG_SERVER_LOG = True # Log events and errors?
DEBUG_SERVER_LOGFILE_EVENT = "./log/log_event"
DEBUG_SERVER_LOGFILE_ERROR = "./log/log_error"

DEBUG_SERVER_WRITE_IMAGE = True # Write received images to disk?
DEBUG_SERVER_WRITE_IMAGE_DIR = "./log/image"
