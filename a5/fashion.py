# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def bar_graph(train, test, class_names):
    counter = {}
    for i in range(len(train)):
        label = train[i]
        if class_names[label] not in counter:
            counter[class_names[label]] = 1
        else:
            counter[class_names[label]] += 1
    
    for i in range(len(test)):
        label = test[i]
        if class_names[label] not in counter:
            counter[class_names[label]] = 1
        else:
            counter[class_names[label]] += 1
    
    num_of_instances = []
    for c in class_names:
        num_of_instances.append(counter[c])
    y_pos = np.arange(len(class_names))
    plt.bar(y_pos,num_of_instances,width=0.4,align='center',alpha=0.5)
    plt.xticks(y_pos,class_names)
    plt.title('number of instances of each class')
    # plt.show()





if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    bar_graph(train_labels,test_labels,class_names)
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)