""" Compute face embeddings with pre-trained network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import src.facenet
import os
import sys
import math

def get_embeddings(images, sess, images_placeholder, phase_train_placeholder, embeddings):
    ''' Compute embeddings for images (arrays) using pre-loaded model
        images: array of shape (n_imgs, img_size, img_size, 3) (RGB)
        sess: TensorFlow session with pre-loaded graph
        embeddings: TenorFlow layer with the embeddings
     '''

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on images')
    

    nrof_images = images.shape[0]
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    
    return emb_array

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Get the paths for the images
            paths = []
            for imagePath in os.listdir(args.face_dir):
                imagePath = os.path.join(args.face_dir, imagePath)
                if os.path.isfile(imagePath):
                    print(imagePath)
                    paths.append(imagePath)

            
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')
            batch_size = args.batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, True, True, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        

            # Save embeddings to disk
            for i in range(nrof_images):
                np.save(paths[i], emb_array[i,:], allow_pickle=False)
                
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('face_dir', type=str,
        help='Path to the data directory containing face patches.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=100)
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
