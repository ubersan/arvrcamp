#!/usr/bin/env python3

import time
import os
import tensorflow as tf
import cv2
import numpy as np
import math
from sklearn.externals import joblib
from scipy import misc

import configuration

import src.detect_face as detect_face
import src.facenet as facenet
from src.compute_embeddings import get_embeddings

class ClassifierFace():

    """Classifier"""


    def __init__(self):

        """Initialize classifier"""

        self.load_models()


    # Load models

    def load_models(self):

        """Load models"""

        print("Classifier: loading models ...")

        self.load_model_haar()
        self.load_model_align()
        self.load_model_facenet()
        self.load_model_svm()

        print("Classifier: loading models ... OK")


    def load_model_haar(self):

        """Loads the haar cascade model"""

        print("Classifier: loading model haar cascade ...")

        self.model_haar = cv2.CascadeClassifier(os.path.join(configuration.HAARCASCADE_MODEL_DIR, 'haarcascade_frontalface_default.xml'))

        print("Classifier: loading model haar cascade ... OK")


    def load_model_align(self, gpu_memory_fraction=1.0):

        """Loads the align model"""

        print("Classifier: loading model align ...")

        with tf.Graph().as_default():

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            session_align = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with session_align.as_default():

                self.model_align_pnet, self.model_align_rnet, self.model_align_onet = detect_face.create_mtcnn(session_align, configuration.MTCNN_MODEL_DIR)

        print("Classifier: loading model align ... OK")


    def load_model_facenet(self):

        """Loads the facenet model"""

        print("Classifier: loading model facenet ...")

        self.model_facenet_g = tf.Graph()
        self.model_facenet_g.as_default().__enter__()
        self.model_facenet_session_tf = tf.Session()
        self.model_facenet_session_tf.__enter__()

        # Load the model
        print('Model directory: %s' % configuration.FACEEMBEDDING_MODEL_DIR)

        meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(configuration.FACEEMBEDDING_MODEL_DIR))

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        facenet.load_model(configuration.FACEEMBEDDING_MODEL_DIR, meta_file, ckpt_file)

        # Get input and output tensors

        self.model_facenet_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.model_facenet_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.model_facenet_phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        self.model_facenet_image_size = self.model_facenet_images_placeholder.get_shape()[1]
        self.model_facenet_embedding_size = self.model_facenet_embeddings.get_shape()[1]

        print("Classifier: loading model facenet ... OK")


    def load_model_svm(self):

        """Loads the svm model"""

        print("Classifier: loading model svm ...")

        self.model_svm = joblib.load(os.path.join(configuration.SVM_MODEL_DIR, 'svm-face.pkl'))
        self.model_svm_classes = self.model_svm.classes_

        print("Classes: {0}".format(self.model_svm_classes))

        print("Classifier: loading model svm ... OK")


    # TODO

    # Classify faces

    def classify(self, image):

        """Classify faces"""

        print("Classifier: classifying ...")

        boxes_haar = self.face_detection_haar_cascade(image)

        if len(boxes_haar) > 0:

            scaled_imgs = []
            bounding_boxes = []

            for (xmin, ymin, xmax, ymax) in boxes_haar:

                scaled, bb = self.align_face(image[ymin:ymax, xmin:xmax])

                if scaled is not None and bb is not None:

                    scaled_imgs.append(scaled)
                    bounding_box = [int(xmin+bb[0]), int(ymin+bb[1]), int(xmin+bb[2]), int(ymin+bb[3])]
                    bounding_boxes.append(bounding_box)

                else:
                    continue

            if len(scaled_imgs) > 0:

                emb = self.compute_embeddings(scaled_imgs)
                prob = self.predict_class_with_svm(emb)
                predictions = []

                for prediction in prob:

                    if max(prediction) > 0.65:

                        ind = prediction.tolist().index(max(prediction))
                        predictions.append(self.model_svm_classes[ind])

                    else:
                        predictions.append("others")

                return bounding_boxes, predictions

            else:

                errorMessage = "unable to predict frame"
                print(errorMessage)
                return None, None
        else:
            return None, None


    def face_detection_haar_cascade(self, image, sF=1.1, minN=4, margin=0.2):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image

        facesFront = self.model_haar.detectMultiScale(
            image=gray,
            scaleFactor=sF,
            minNeighbors=minN,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        #print("Found {0} frontal faces!".format(len(facesFront)))
        
        boxes = []

        if len(facesFront) > 0:

            for (x, y, w, h) in facesFront:

                xmargin = int(margin * image.shape[1])
                ymargin = int(margin * image.shape[0])
                xmin = max(0, x-xmargin)
                xmax = min(image.shape[1], x+w+xmargin)
                ymin = max(0, y-ymargin)
                ymax = min(image.shape[0], y+h+ymargin)
                boxes.append([xmin, ymin, xmax, ymax])
                
        return boxes


    def align_face(self, face, image_size=182, margin=44):

        minsize = 50 # minimum size of face
        threshold = [ 0.7, 0.6, 0.6 ]  # three steps's threshold
        factor = 0.709 # scale factor
        
        try:
            img = face
        except (IOError, ValueError, IndexError) as e:
            errorMessage = 'Unable to align image'
            print(errorMessage)
        else:

            if img.ndim<2:
                print('Unable to align image')

            if img.ndim == 2:
                img = facenet.to_rgb(img)

            img = img[:,:,0:3]

            bounding_boxes, _ = detect_face.detect_face(img, minsize, self.model_align_pnet, self.model_align_rnet, self.model_align_onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces>0:

                det = bounding_boxes[:,0:4]
                img_size = np.asarray(img.shape)[0:2]

                if nrof_faces>1:

                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det = det[index,:]

                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                return scaled, bb

            else:
                print('Unable to align image')
                return None, None


    def compute_embeddings(self, imgs, batch_size = 100):

        nrof_images = len(imgs)
        nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.model_facenet_embedding_size))

        for i in range(nrof_batches):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)

#            paths_batch = paths[start_index:end_index]

            images = self.load_data(imgs, False, False, self.model_facenet_image_size, True)
            emb_array[start_index:end_index,:] = get_embeddings(images, self.model_facenet_session_tf, self.model_facenet_images_placeholder, self.model_facenet_phase_train_placeholder, self.model_facenet_embeddings)

        return emb_array


    def predict_class_with_svm(self, emb_array):

        emb_array = np.array(emb_array)
        prob = self.model_svm.predict_proba(emb_array)
        return prob


    def load_data(self, imgs, do_random_crop, do_random_flip, image_size, do_prewhiten=True):

        nrof_samples = len(imgs)
        images = np.zeros((nrof_samples, 160, 160, 3))

        for i in range(nrof_samples):

            img = imgs[i]
            if img.ndim == 2:
                img = to_rgb(img)
            if do_prewhiten:
                img = facenet.prewhiten(img)
            img = facenet.crop(img, do_random_crop, image_size)
            img = facenet.flip(img, do_random_flip)
            if (image_size, image_size) != img.shape[:2]:
                img = misc.imresize(img, (image_size, image_size))
            images[i,:,:,:] = img

        return images
