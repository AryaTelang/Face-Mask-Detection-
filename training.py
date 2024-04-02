from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

directory=r"FaceMaskDetection\dataset"
categories=["with_mask","without_mask"]

data=[]#image arrays
labels=[]#labels about with mask or without

for category in categories:
    path=os.path.join(directory,category)
    #list down all images in path
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        #using keras we change img to 224,224
        image=load_img(img_path,target_size=(224,224))
        #convert img to array
        image=img_to_array(image)
        #mobilenets
        image=preprocess_input(image)
        data.append(image)
        labels.append(category)#in alphabets
#one hot encoding to get into 0s 1s
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#construct many more images for training by slightly changing 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#pretrained weights
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
#constructs a CNN model for binary classification on top of the MobileNetV2 architecture
headModel = baseModel.output
'''
Pooling, 
in the context of convolutional neural networks (CNNs), 
is a technique used to reduce the spatial dimensions (width and height) of the feature map while 
retaining the important information.'''
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
#best for images=>relu
headModel = Dense(128, activation="relu")(headModel)
#orevent overfitting
headModel = Dropout(0.5)(headModel)
#to get two outputs
headModel = Dense(2, activation="softmax")(headModel)

#we have a base model which is pretrained, it is a deep neural network
# In transfer learning, instead of training a model from scratch, you leverage the knowledge learned by the base model on a similar task and fine-tune it for your specific task.
#The head refers to the additional layers that are added on top of the base model. These layers are specific to the task you are trying to solve. 

model = Model(inputs=baseModel.input, outputs=headModel)
#freeze layers
for layer in baseModel.layers:
	layer.trainable = False
#adam is best for image methods
opt = Adam(INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


predIdxs = model.predict(testX, batch_size=BS)
#print(classification_report(testY.argmax(axis=1), predIdxs,
#	target_names=lb.classes_))

model.save("mask_detector.h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")