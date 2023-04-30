from Functions import *
from sklearn.model_selection import train_test_split
import os
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np

## STep 1
path='myData'
data=importDataInfo(path)

## Step 2 Data balancing
data=balanceData(data,display=False)

## Step 3 Loaddata
imagesPath,steering=loadData(path,data)
#print(imagesPath[0],steering[0])


## Step4
xtrain,xval,ytrain,yval=train_test_split(imagesPath,steering,test_size = 0.2, random_state=5)

print("Train",len(xtrain))
print("Test",len(xval))


##Step 5



##Step 6

##Step 7

##Step 8
model = createModel()
model.summary()

##Step 9
history= model.fit(batchGen(xtrain, ytrain, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batchGen(xval, yval, 100, 0),
                                  validation_steps=200 )


##Step 10

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim(0,1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()