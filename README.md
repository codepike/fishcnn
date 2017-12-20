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
$train.sh

Test a model with an image or directory containing images
$python cnn/predict_1.py --checkpoint_dir checkpoint --path ../fish_dataset/fish --image path/to/image

Package a model
$python cnn/package_model.py --checkpoint_dir ./checkpoint --model_dir model

Update a model to gs
$gstuil cp ./model/* gs://fishcnn/fcnn6

Create a version
$gcloud ml-engine versions create fcnn6 --model=fcnn --origin=gs://fishcnn/fcnn6 --staging-bucket gs://fishcnn --runtime-version=1.2

