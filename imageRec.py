#Please note that this is an initial experimentation. It is the same as the ML basics with Keras tutorial on TF with
#slight modifications and notes. 

#Key:
#[!] = Unclear on certain aspects (Followed by list of concepts not yet grasped)
#[!!] = No idea what is going on here. (Followed by list of concepts not yet grasped)
#[x] = Well understood 

#Hyperparams:
#Epochs - 80




#==Important Packages== [x]
# TensorFlow and tf.keras
import tensorflow as tf
#Keras is an API for TF that makes doing work with it easier. 
from tensorflow import keras

# ==Aux libraries== [x]
#Numpy is a package that allows for advanced vectors
import numpy as np
#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt

#==DB and labels== [x]
#Database is from mnist, of various clothes
fashionDb = keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fashionDb.load_data()

#There are 10 labels. These are the objects associated with them. Each image in the DB
#has a label of 0 to 9, which corresponds to the index of the name in the array.
objectNames = ['T-shirt', 'Shorts', 'Pullover', 'Dress', 'coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'boot']



#==Preprocessing the data.== [x]
#Sets the range of values to a range of 0 to 1. 
trainImages = trainImages / 255.0
testImages = testImages / 255.0

#Sets the figures to 10x10
plt.figure(figsize=(10,10))


#==Creating the layers.== [!] - "relu"
model = keras.Sequential([
    #The first (input) layer (keras.layers.Flatten) in the network
    #takes the 2d array of 28x28 pixels shared by each of the
    #images to a 1d array (28 * 28 = 784) 
    keras.layers.Flatten(input_shape=(28,28)),

    #There are two more layers in the network. These are fully connected (dense)
    #layers. The first layer has 128 nodes/neurons (hidden). "relu" indicates the application of the Applies the rectified linear unit activation function
    #The second returns a Logits(??) array. Each entry in the array has a score that indicates one of the 10 objects (output). 

    #Note: Activation functions are mathematical equations that determine the output of a neural network.
    #The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, 
    #based on whether each neuron’s input is relevant for the model’s prediction. Activation functions also help normalize the
    #output of each neuron to a range between 1 and 0 or between -1 and 1.
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation=None)
])

# ==Compiling== [!!] - "Optimizer," "loss function," "metrics"

#Before running, the net needs a few more settings. This is done in the compile step
#Sets the optomizer, loss function, and metrics.
#The Optimizer is how the model is updated based on the data it sees and its loss function. 
#The loss function is a method of evaluating how well specific algorithm models the given data. 
#Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#==Training the model== [x]
#To start training, call model.fit. Params are features, labels, and epochs.
model.fit(trainImages, trainLabels, epochs=10)

#==Evaluate Accuracy== [x]
#Gets the loss and accuracy of the model
testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=2)
print('\nTest accuracy:', testAcc)

#==Making predictions== [!!] - "Logits," "softmax," "predict method"
#With the model trained, you can use it to make predictions about some images. 
#The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#Makes a list of predictions(?)
predictions = probModel.predict(testImages)

#==Graphing predictions== [!] - not familiar with matlib usage
def plotImage(i, predictArr, trueLabel, img):
  trueLabel, img = trueLabel[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predictedLabel = np.argmax(predictArr)
  if predictedLabel == trueLabel:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(objectNames[predictedLabel],
                                100*np.max(predictArr),
                                objectNames[trueLabel]),
                                color=color)

def plotValueArray(i, predictArr, trueLabel):
  trueLabel = trueLabel[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictArr, color="#777777")
  plt.ylim([0, 1])
  predictedLabel = np.argmax(predictArr)

  thisplot[predictedLabel].set_color('red')
  thisplot[trueLabel].set_color('blue')


#==Graph a couple examples== [x]
#rows = 5
#cols = 3
#num_images = rows*cols
#plt.figure(figsize=(2*2*cols, 2*rows))
#for i in range(num_images):
#  plt.subplot(rows, 2*cols, 2*i+1)
#  plotImage(i, predictions[i], testLabels, testImages)
#  plt.subplot(rows, 2*cols, 2*i+2)
#  plotValueArray(i, predictions[i], testLabels)
#plt.tight_layout()
#plt.show()

#==Using the trained model== [!]
# Add the image to a batch where it's the only member.
# Batches are a collection of examples.
i = 40
img = (np.expand_dims(testImages[i], 0))

#Makes a single prediction
predictionsSingle = probModel.predict(img)

#Plots a graph of the tested image
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plotImage(i, predictionsSingle[0], testLabels, testImages)
plt.subplot(1,2,2)
plotValueArray(i, predictionsSingle[0],  testLabels)
_ = plt.xticks(range(10), objectNames, rotation=45)
plt.show()


np.argmax(predictionsSingle[0])

