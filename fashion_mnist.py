# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 16:45:07 2018

@author: Kriti Gupta
"""


from keras.models import Sequential, Model
from keras.layers import Input,Dense,Activation,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y2indicator(Y):
    N=len(Y)
    K=len(set(Y))
    I=np.zeros((N,K))
    I[np.arange(N),Y]=1
    return I

data=pd.read_csv(r"fashion-mnist_train.csv")
data=data.values
np.random.shuffle(data)

X=data[:,1:].reshape(-1,28,28,1)/255.0
Y=data[:,0].astype(np.int32)

K=len(set(Y))
Y=y2indicator(Y)

i=Input(shape=(28,28,1))
x=Conv2D(filters=32,kernel_size=(3,3))(i)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D()(x)

x=Conv2D(filters=64,kernel_size=(3,3))(i)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D()(x)

x=Flatten()(x)
x=Dense(units=100)(x)
x=Activation('relu')(x)
x=Dropout(0.2)(x)
x=Dense(K)(x)
x=Activation('softmax')(x)

model=Model(inputs=i,outputs=x)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
r=model.fit(X,Y,validation_split=0.33,epochs=15,batch_size=32)
print("Returned:",r)
print(r.history.keys())

plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'],label='acc')
plt.plot(r.history['val_acc'],label='val_acc')
plt.legend()
plt.show()

data1=pd.read_csv(r"fashion-mnist_test.csv")
data1=data1.values
X1=data1[:,1:].reshape(-1,28,28,1)/255.0
Y1=data1[:,0].astype(np.int32)
l1=[]
for i in range(0,10000):
    l1.append(Y1[i])
    
predictions=model.predict(X1,batch_size=32,verbose=1)
l=[]
for i in range(0,10000):
    value=predictions[i,:]
    ind=np.argmax(value)
    l.append(ind)
    
cnt=0    
for i in range(0,10000):
    if l[i]==l1[i]:
        cnt=cnt+1
accuracy=cnt/10000.0
print(accuracy*100)