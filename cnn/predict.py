from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import argparse
from scipy import misc
from googleapiclient import errors
# Time is for waiting until the request finishes.
import time
import json

projectID = 'projects/{}'.format('bluefish-166900')
modelName = 'fcnn'
modelID = '{}/models/{}'.format(projectID, modelName)
versionName = 'fcnn6'
versionDescription = 'version_description'
trainedModelLocation = 'gs://linear_model'

from tensorflow.examples.tutorials.mnist import input_data


# credentials = GoogleCredentials.get_application_default()
# ml = discovery.build('ml', 'v1', credentials=credentials)


# requestDict = {'name': modelName,
#     'description': 'Another model for testing.'}

# request = ml.projects().models().create(parent=projectID,
#                             body=requestDict)

# try:
#     response = request.execute()

#     # Any additional code on success goes here (logging, etc.)

# except errors.HttpError as err:
#     # Something went wrong, print out some information.
#     print('There was an error creating the model.' +
#         ' Check the details:')
#     print(err._get_reason())

#     # Clear the response for next time.
#     response = None

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('ml', 'v1', credentials=credentials)
    name = '{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    # print(instances)
    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']



# f = 1.0
# a = {'x': f}


# data_dir = './mnist'
# mnist = input_data.read_data_sets(data_dir)

# batch = mnist.test.next_batch(10)

# for i in range(10):
#     a = {'x': batch[0][i].tolist()}

# print(batch[0][0])
# print(batch[1][0])
#

# from PIL import Image
import numpy
import numpy as np

def run1(filename):
    print(filename)
    img = misc.imread(filename)
    img = misc.imresize(img, (64,64))
    img = img.reshape(1, 12288).astype(np.float32)
    inst = []
    inst.append(img.tolist())
    r = predict_json(projectID, modelName, inst, version="fcnn6")

    print(r)

# def run(input):
#     filename = input
#     im = Image.open(filename)
#
#     im = im.convert('L')
#
#     im = im.resize((28,28))
#
#     mat = numpy.array(im)
#
#     mat = 255-mat
#
#     mat = mat.astype(numpy.float32)/255.0
#
#     print(mat.shape)
#
#
#     inst = []
#     inst.append(mat.tolist())
#     r = predict_json(projectID, modelName, inst, version="fcnn4")
#
#     print(r)
# print(batch[1][i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename',
        help='The checkpoint file to build a model',
        required=True
    )


    config = parser.parse_args()

    run1(config.filename)
