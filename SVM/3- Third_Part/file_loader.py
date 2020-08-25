from os import listdir
from matplotlib import image
from numpy import asarray
import matplotlib.pyplot as plt
import numpy as np
images = []
labels = []
for filename in listdir('Dataset/W/'):
    img_data = image.imread('Dataset/W/' + filename)
    images.append(asarray(img_data))
    labels.append('W')

for filename in listdir('Dataset/S/'):
    img_data = image.imread('Dataset/S/' + filename)

    images.append(asarray(img_data))
    labels.append('S')

for filename in listdir('Dataset/7/'):
    img_data = image.imread('Dataset/7/' + filename)

    images.append(asarray(img_data))
    labels.append('7')

for filename in listdir('Dataset/3/'):
    img_data = image.imread('Dataset/3/' + filename)

    images.append(asarray(img_data))
    labels.append('3')

for filename in listdir('Dataset/2/'):
    img_data = image.imread('Dataset/2/' + filename)

    images.append(asarray(img_data))
    labels.append('2')

# print(images)

images = np.array(images)
labels = np.array(labels)

p = np.random.permutation(len(images))

images=images[p]
labels=labels[p]


np.savez('persian_lpr.npz' , images=images , targets=labels)

dataset = np.load('persian_lpr.npz')

print(dataset['images'])

plt.imshow(dataset['images'][1])
print(dataset['targets'][1])
plt.show()
