# fishcnn
The project is trying to identify fish species from images. The model
is convolution neural network(CNN).  

Notes: 
Data format
Images are 64x64 RGB.

Training:
Train locally:
$python cnn/main.py --train --dataset datafile.tfrecord --epoch 10
$gcloud ml-engine local train  --module-name=cnn.main -- --train --dataset=datafile.tfrecrod --epoch 10

Train on a cloud:
$gcloud ml-engine train  --module-name=cnn.main -- --train --dataset=datafile.tfrecrod --epoch 10