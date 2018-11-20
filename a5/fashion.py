# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.eager as tfe

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

    #For question 2c
    # bar_graph(train_labels,test_labels,class_names)

    #For question 2c(ii) to 2g
    # Here I do not do data seperation because they are pre-seperated already according to Elsa Riachi
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

    hist = model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_split=0.3)

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    iterations = np.arange(10)

    plt.figure(1,figsize=(7,5))
    plt.plot(iterations,train_loss)
    plt.plot(iterations,val_loss)
    plt.xlabel('number of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.legend(['train','val'])

    plt.figure(2,figsize=(7,5))
    plt.plot(iterations,train_acc)
    plt.plot(iterations,val_acc)
    plt.xlabel('number of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.legend(['train','val'])
    plt.show()


    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)