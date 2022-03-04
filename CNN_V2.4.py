import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from astropy.io import fits
from astropy.io import ascii
from numpy import savetxt
import keras_tuner as kt

#important params
folder_path =  "/c4/wfh842"
write_path = "/c4/wfh842/CNNmodelV2.4"

#Get data
img_size=200
t=ascii.read(folder_path+'/catalog95.csv')
file_name_array = os.listdir(folder_path+'/simimg')
length=len(file_name_array)
train_data=[]
train_labels=[]
val_data=[]
val_labels=[]
test_data=[]
test_labels=[]
test_ID=[]
for i in range(length):
  file_path=folder_path +'/simimg' +"/"+ file_name_array[i]
  hdu = fits.open(file_path,memmap=False)
  data = hdu[0].data
  data=data/np.max(data)
  ID=int(file_name_array[i][:-5])
  #print(str(i)+'/'+str(length-1))
  for j in range(len(t['ID'])):
            if round(t['ID'][j])==ID:
                label=round(t['Label'][j])
                break
  if i%6==0:
    val_data.append(data)
    val_labels.append(label)
  elif i%7==0:
    test_ID.append(ID)
    test_data.append(data)
    test_labels.append(label)
  else:
    train_data.append(data)
    train_labels.append(label)
  hdu.close()
  del data
train_ds = tf.data.Dataset.from_tensor_slices((np.array(train_data), np.array(train_labels)))
test_ds = tf.data.Dataset.from_tensor_slices((np.array(test_data), np.array(test_labels)))
val_ds = tf.data.Dataset.from_tensor_slices((np.array(val_data), np.array(val_labels)))
class_names = ('Below','Above')
print(len(test_ID))


#Tune model
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_ds = val_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)
num_classes = 2
#maxvals=[np.max(val_data),np.max(train_data),np.max(test_data)]
#maxval=np.max(maxvals)


def build_model(hp):
    a=units = hp.Int("units", min_value=2, max_value=16, step=2)
    b=units = hp.Int("units", min_value=2, max_value=4, step=1)
    d=units = hp.Int("units", min_value=64, max_value=256, step=64)
    model = Sequential([
      layers.Rescaling(1, input_shape=(img_size, img_size,1)),
      layers.Conv2D(a, b, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(2*a, b, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(4*a, b, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(d, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=write_path,
                     project_name='tuner')
tuner.search_space_summary()
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
epochs=100
tuner.search(np.array(train_data), np.array(train_labels), epochs=epochs, validation_data=(np.array(val_data), np.array(val_labels)), callbacks=[es])
#tuner.search(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[es])

model = tuner.get_best_models()[0]
epochs_range = range(epochs)

#Fit Model
model.summary()
history = model.fit(
  x=np.array(train_data), 
  y=np.array(train_labels),
  validation_data=(np.array(val_data), np.array(val_labels)),
  epochs=epochs,
  callbacks=[es])
model.save(write_path+'/model')
np.save(write_path+'/traininghistory.npy',history.history)

#Plot results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['loss']))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy fraction')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss fraction')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(write_path+'/AccuracyLossPlot.png')

#Run test data
correct=0
redx=[]
redy=[]
greenx=[]
greeny=[]
for i in range(len(test_data)):
  img_array = np.array(test_data[i])
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  ID=test_ID[i]
  for j in range(len(t['ID'])):
            if round(t['ID'][j])==ID:
                label=round(t['Label'][j])
                sSFR=t['SSFR'][j]
                if sSFR<10**(-14):
                    sSFR=10**(-14)
                M=t['M_s'][j]
                break
  if test_labels[i]==1:
    imgclass='Above'
  else:
    imgclass='Below'
  print(
      "Prediction: {} with a {:.2f} percent confidence, Actual= {}"
      .format(class_names[np.argmax(score)], 100 * np.max(score),imgclass)
  )
  if class_names[np.argmax(score)]==imgclass:
    correct=correct+1
    greenx.append(M)
    greeny.append(sSFR)
  else:
    redx.append(M)
    redy.append(sSFR)
print("Score="+str(correct)+'/'+str(len(test_ID)))
#Save important test data
#savetxt(write_path+'/test_data.csv', np.array(test_data), delimiter=',')
savetxt(write_path+'/test_ID.csv', np.array(test_ID), delimiter=',')
f = open(write_path+"/score.txt", "w")
#txt="a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)
#f.write(txt)
#f.write('\n')
f.write("Score="+str(correct)+'/'+str(len(test_ID)))
f.write('\n')
f.write("Training Accuracy="+str(acc[len(history.history['loss'])-6]))
f.write('\n')
f.write("Training Loss="+str(loss[len(history.history['loss'])-6]))
f.write('\n')
f.write("Epochs="+str(len(history.history['loss'])-5))
f.close()
#Plot test results
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(greenx,greeny,'g.',label='Correct')
ax.plot(redx,redy,'r+',label='Incorrect')
ax.axhline(y=10**-11, color='k', linestyle='-')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Stellar Mass M_0')
ax.set_ylabel('SSFR yr^-1')
ax.legend()
fig.savefig(write_path+'/TestResults.png')
