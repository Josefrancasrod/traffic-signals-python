import tensorflow as tf
import os
import skimage.data as imd
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.color import rgb2gray

labels = []
images = []

def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
                                if os.path.isdir(os.path.join(data_directory, d))]
    # print(dirs)
    

    for d in dirs:
        label_dir = os.path.join(data_directory,d)
        file_names = [os.path.join(label_dir,f)
                    for f in os.listdir(label_dir)
                    if f.endswith(".ppm")]
        # print(label_dir)
        # print(file_names)

        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))

    return images, labels

main_dir = "./"
train_data_dir = os.path.join(main_dir, "Training")
test_data_dir = os.path.join(main_dir, "Testing")
images, labels = load_ml_data(train_data_dir)
# labels = load_ml_data(train_data_dir)

# print(type(images))
# print(type(labels))

images = np.array(images)
labels = np.array(labels)

# print(images.size, images.ndim)  
# print(labels.size, labels.ndim)

len(set(labels))
plt.hist(labels, len(set(labels)))


rand_signs = random.sample(range(0, len(labels)), 6)
print(rand_signs)

for i in range(len(rand_signs)):
    temp_im = images[rand_signs][i]
    plt.subplot(1, 6, i+1)
    
    plt.imshow(temp_im)
    plt.subplots_adjust(wspace = 0.5)
# plt.show()

unique_labels = set(labels)
plt.figure(figsize=(16,16))
i=1
for label in unique_labels:
    temp_im = images[list(labels).index(label)]
    plt.subplot(8,8,i)
    plt.axis("off")
    plt.title("Clase {0} ({1})".format(label, list(labels).count(label)))
    i += 1
    plt.imshow(temp_im)
plt.show()

from skimage import transform
w = 999
h = 999

for image in images:
    #1 ES ALTURA 0 ES ANCHO
    if image.shape[0] < h:
        h = image.shape[0]

    if image.shape[1] < w:
        w = image.shape[1]
print("Tamaño minimo de las imagens_ {0} X {1}".format(h,w))

images30 = [transform.resize(image, (30,30)) for image in images]
images30 = np.array(images30)
images30 = rgb2gray(images30)

#definir red neuronal

#Parametros de entrada                         imagenes 30x30
x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])

y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y , logits=logits))

train_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

final_pred = tf.argmax(logits,1)
accurancy = tf.reduce_mean(tf.cast(final_pred, tf.float32))

#Entrenamiento
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(600):
    _, accurancy_val = sess.run([train_opt, accurancy],feed_dict={
                                                            x: images30,
                                                            y: list(labels)
                                                        })

    if i%50 == 0:
        print("Eficacia de la red neuronal", accurancy_val)
    

sample_idx = random.sample(range(len(images30)),30)
sample_images = [images30[i] for i in sample_idx]
sample_labes = [labels[i] for i in sample_idx]

prediction = sess.run([final_pred], feed_dict={x:sample_images})[0]

plt.figure(figsize=(16,20))

for i in range(len(sample_images)):
    real = sample_labes[i]
    predi = prediction[i]
    plt.subplot(10, 4, i+1)
    plt.axis("off")
    color = "green" if real == predi else "red"
    plt.text(32,15, "Real: {0} \n Pediccion: {1}".format(real, predi), fontsize=14, color = color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()    
