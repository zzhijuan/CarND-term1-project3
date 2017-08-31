
# coding: utf-8

# In[50]:

#load data tuples to python
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

lines = []

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[51]:

#load all images for histogram analysis
images = []
measurements = []
for line in lines:
    
    if line[0] == 'center':
        continue
    if float(line[6]) < 0.1:
        continue

    image_path = './IMG/' + line[0].split('/')[-1]
    image = cv2.imread(image_path) #BGR
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement) 
    
    image_path = './IMG/' + line[1].split('/')[-1]
    image = cv2.imread(image_path) #BGR
    images.append(image)
    measurement = float(line[3]) + 0.25
    measurements.append(measurement) 

    image_path = './IMG/' + line[2].split('/')[-1]
    image = cv2.imread(image_path) #BGR
    images.append(image)
    measurement = float(line[3]) - 0.25
    measurements.append(measurement)     


# In[81]:

plt.figure(num = 1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(measurements, 'b-',  [0, 30000], [0, 0], 'r--')


# In[76]:

#display image histogram function
def histImages(measurements):
    nb_bins = 30
    hist, bins = np.histogram(measurements, nb_bins)
    width = 0.5 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2.0
    mean_sample = len(measurements)/nb_bins
    plt.bar(center, hist, align = 'center', width = width)
    #plt.plot((np.min(measurements), np.max(measurements)),(mean_sample, mean_sample), 'r-' )


# In[77]:

#Display image histogram
plt.figure()
histImages(measurements[:40000])


# In[55]:

#Remove strong bias samples function
def removeList(images, measurements):
    nb_bins = 30
    hist, bins = np.histogram(measurements, nb_bins)
    keep_list = []
    mean_sample = len(measurements)/nb_bins 
    
    for i in range(nb_bins):
        if hist[i] < mean_sample:
            keep_list.append(1.0)
        else:
            keep_list.append(mean_sample/hist[i])
    
    remove_list = []
    for i in range(len(measurements)):
        for j in range(nb_bins):
            if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
                if np.random.rand() > keep_list[j]:
                    remove_list.append(i)
    new_images = np.delete(images, remove_list, axis=0)
    new_measurements = np.delete(measurements, remove_list)
    return new_images, new_measurements, remove_list


# In[56]:

#flip images function
def flip_image(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        if abs(measurement) > 0.35:
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement*(-1.0))
    x_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    return x_train, y_train


# In[78]:

#Filter images (only the first 40000 samples are used to speed up)
new_images, new_measurements, remove_list = removeList(images[:40000], measurements[:40000])

#Display histogram after remove
plt.figure()
histImages(new_measurements)

#Display histogram after flipping (for internal checking only, filp will be applied to the new_images/measurements in generator)
temp_images, temp_measurements = flip_image(new_images, new_measurements)
plt.figure()
histImages(temp_measurements)


# In[58]:

#convert RGB to YUV & resize images
def RGB2YUV(image):
    # convert to YUV color space (to be the same as in nVidia)
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return new_img


# In[59]:

#random light change & random shadow function
def random_light(image):
    new_image = image.astype(float)
    value = np.random.randint(low = -30, high = 30)
    if value >= 0:
        mask = (image[:,:,0] + value) > 255
    if value < 0:
        mask = (image[:,:,0] + value) < 0
    new_image[:,:,0] += np.where(mask, 0, value)
    
    #random shadow
    height,width = new_image.shape[0:2]
    splitpoint = np.random.randint(0,width)
    scale = np.random.uniform(0.3,1)
    if np.random.rand() > .5:
        new_image[:,0:splitpoint,0] *= scale
    else:
        new_image[:,splitpoint:width,0] *= scale
        
    image = new_image.astype('uint8')
    return image   


# In[80]:

#check one random image and comapre before & after preprocessing
image_test = new_images[1000]
angle_test = new_measurements[1000]

print(np.shape(new_images[1000]))
plt.figure()
plt.imshow(image_test, cmap = 'gray')

plt.figure()
plt.imshow(random_light(RGB2YUV(image_test[60:135,:,:])), cmap = 'gray')



# In[61]:

#generator and apply preprocessing functions internally 
import sklearn

def generator_train(images, measurements, batch_size):
    images, measurements = sklearn.utils.shuffle(images, measurements)
    num_samples = len(images)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            X_train = []
            y_train = []
            samples = images[offset:offset+batch_size]
            angles  = measurements[offset:offset+batch_size]
            for sample, angle in zip(samples, angles):
                X_train.append(random_light(RGB2YUV(sample)))
                y_train.append(angle)
            
            final_x_samples, final_y_samples = flip_image(X_train, y_train)
            yield sklearn.utils.shuffle(final_x_samples, final_y_samples)
            X_train = []
            y_train = []


# In[62]:

def resize_comma(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, [66, 200])


# In[63]:

#split training and validation samples tuples (10%)
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(new_images, new_measurements, test_size=0.1, random_state=42)


# In[65]:

#generate training and validation samples (preprocessing applied)
#compile and train the model using the generator function
train_generator = generator_train(train_x, train_y, batch_size=128)
validation_generator = generator_train(val_x, val_y, batch_size=128)


# In[66]:

#Apply Convet model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
import tensorflow as tf

model = Sequential()

#resize image otherwise won't be loaded by drive.py
model.add(Cropping2D(cropping=((60, 25), (0, 0)),input_shape=(160, 320, 3)))


# Resize the data
model.add(Lambda(resize_comma))

# Normalize
model.add(Lambda(lambda x: x/255.0 - 1.0))

# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dropout(0.50))
    
# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dropout(0.50))

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, kernel_regularizer=l2(0.001)))
model.add(ELU())

# Add a fully connected output layer
model.add(Dense(1))


# In[73]:

#train model  
model.compile(loss = 'mse', optimizer='adam')
batch_size = 128  
model.fit_generator(train_generator, steps_per_epoch= len(train_x)/batch_size, 
                    validation_data=validation_generator, validation_steps=len(val_x)/batch_size, 
                    epochs=12)


# In[74]:

#save tuned model as h5 file
model.save('model_5.h5')

