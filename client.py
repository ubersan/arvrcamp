#/usr/bin/env python3

"""Client template"""

import hashlib
import pickle
import numpy as np
import cv2
import requests
import logger


def send_image_to_server(filename):

    """Send image to server"""

    # Load image from disk and convert to numpy array

    image = cv2.imread(filename, 1)
    image_matrix = np.asarray(image)

    # Compute MD5 hash and log it

    hash_md5 = hashlib.md5()
    hash_md5.update(image_matrix)
    hash_image_matrix = hash_md5.hexdigest()
    logger.log_event("Client: sent image matrix MD5 hash is: " + hash_image_matrix)

    # Serialize image matrix (marked below with horizontal rules)
    # NOTE: THIS SERIALIZATION NEEDS TO MATCH THE SERVER'S DESERIALIZATION

    # ---------------------------------------------------------------------

    # Pickle to byte string
    image_pickled = pickle.dumps(image_matrix, protocol=3)

    # ---------------------------------------------------------------------

    # Send image to server

    url = "http://52.28.196.4:5000/api/detect/faces"
    # Set headers to send as binary stream
    headers = {'content-type': 'application/octet-stream'}
    response = requests.post(url, data=image_pickled, headers=headers)

    print(response.json())


def main():

    """Main"""

    send_image_to_server("test/data/face.jpg")


if __name__ == "__main__":

    main()
