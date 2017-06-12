# Readme

This is a server backend for face classification to be used via API from e.g. a Hololens client.

# Requirements

- Python 3
- Flask
- Tensorflow
- OpenCV
- Numpy
- Scikit-learn
- Scipy
- Requests (Python package)

# Run

### Get Project

- `git clone https://username@bitbucket.zuehlke.com/scm/daana/hololens_backend.git path/to/local`
- (`git checkout development` check out development branch to begin development work!)
- (Use default checked out master branch for deployment)

### Run Server

- `python3 server.py`
- This will take a while, since face detection and classification models are loaded
- When Flask reports `Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)` the server is ready to receive images

### Run Test Client

- `python3 client.py`
- This will load a test image, serialize it, send it to the server via API, and print the response
- (This should print out `{'bounding_boxes': [[247, 132, 472, 412]], 'predictions': ['others']}` for the `face.jpg` test image, i.e. a face is found, but it could not be recognized)

# Develop, Collaborate

### Branch Structure

- `master` branch is the stable branch and should always be in a working and stabel state
- `development` branch is for current development
- Use feature branches to test things out and implement additional features, then merge into `development` for integration/testing, then merge into `master` for deployment

### Project Structure

- `server.py` implements the server in Flask
- `configuration.py` defines various settings, server port, model directories, logging behaviour
- `logger.py` implements an event and error logger
- `classifierFace.py` implements face detection and face classification
- `client.py` implements an example client to communicate with the server
- `model/` contains face detection and classification models used by the face classifier
- `log/` contains event and error logs, as well as logged image files
- `test/` contains test files

### Client / Server Communication

- In the example Python client to Python server communication, the image is serialized to a byte string using Python's pickle, and unpickled on the server side.
- To customize how images are sent from a client to the server, the corresponding sections are marked both in `client.py` as well as in `server.py` (*it is important that serialization and deserialization match exactly, such that the image is understood by the face classification module.*)
- Ensure that MD5 hashes of image data on client and server are equal (see code)

### Flask

- Flask runs in non-threaded mode, i.e. requests are processed one-by-one
- Debug mode is turned off such that the server accepts connections from outside localhost

### Logging

- The server supports some logging, mainly event logging and error logging, using the `logger.py` module
- The server can also write every single image it receives to disk, with a timestamp

# Environment

### Deployment (Linux)

- TODO

### Development (MacOS)

- Install XCode
- Install OpenCV
  - `brew tap homebrew/science`
  - `brew install opencv3 --without-python --with-python3 --with-contrib`
  - `echo /usr/local/opt/opencv3/lib/python3.6/site-packages >> /usr/local/lib/python3.6/site-packages/opencv3.pth`
  - (In case of development in a virtual environment) `echo /usr/local/opt/opencv3/lib/python3.6/site-packages >> ./env/lib/python3.6/site-packages/opencv3.pth`
- Install Python dependencies
- (Optional: set up virtual environment)

### Development (Windows)

- TODO

# API

### Detect

##### Faces

- URL: `http://ip:port/api/detect/faces`
- Purpose: detect faces in an image and respond with face locations as well as face classificationsmodels
- State: implemented
- Request: see `client.py`

### Admin

##### Reload

- URL: `http://ip:port/api/admin/reload`
- Purpose: send a request to the server in order to reload models
- State: template for request handler implemented
- Request: `curl --header "Content-Type: application/json" -X POST --data '{"command":"1"}' http://localhost:5000/api/admin/reload`
