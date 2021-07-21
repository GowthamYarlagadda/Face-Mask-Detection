# Face-Mask-Detection

In this project, Using libraries from keras API and tensorflow, the training of image datasets, which were retrieved by using Keras ImageDataGenerator API, building of Convulational neural network (CNN) was accomplished.

This project was done on locally installed tensorflow with gpu support.

## Objective
- To make your own CNN binary image classifier which can distinguish whether the person is wearing a surgical mask or not in live camera feed.


## Components of Code 

- Importing Modules
- Importing train and test datasets through Keras ImageDataGenerator API
- Checking our class labels
- Defining our CNN Model Layers
- Specifying Optimizers and loss functions
- Training the Model
- saving our model as .h5 file for instant access
- Mask detection for individual pictures
- Mask Detection for live camera feed



### Operation:
-->This code requires runs only on local jupyter notebook and tensorflow with gpu support(CUDA,cuDNN's)

-->Modify the path for xml file and voila!

-->Sometimes tweaking scale factor of detectMultiScale() will help If the code doesn't work or run smooth. (imp: scalefactor>1 eg:1.05, 1.15, 1.2)





| Files | sources |
| ------ | ------ |
| Face mask detection dataset | [https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset][PlDb] |
| Tensorflow(GPU) Documentation | [https://www.tensorflow.org/install/gpu][PlGh] |






   [PlDb]: <https://www.kaggle.com/ashishjangra27/face-mask-12k-images-datasett>
   [PlGh]: <https://www.tensorflow.org/install/gpu>
