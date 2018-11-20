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

def train(model,batch_size,epochs,learning_rate):
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    hist = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.3)

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    iterations = np.arange(epochs)

    plt.figure()
    plt.plot(iterations,train_loss,label='train')
    plt.plot(iterations,val_loss,label='val')
    plt.xticks(range(1,epochs+1))
    plt.xlabel('number of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.legend()
    plt.savefig('./q2_data/loss_batch_{}_learning_{}.png'.format(batch_size,learning_rate))

    plt.figure()
    plt.plot(iterations,train_acc,label='train')
    plt.plot(iterations,val_acc,label='val')
    plt.xticks(range(1,epochs+1))
    plt.xlabel('number of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.legend()
    plt.savefig('./q2_data/acc_batch_{}_learning_{}.png'.format(batch_size,learning_rate))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    #For question 2c
    # bar_graph(train_labels,test_labels,class_names)

    #For question 2c(ii) to 2g
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0],28,28,1)
    test_images = test_images.reshape(test_images.shape[0],28,28,1)

    # model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(128, activation=tf.nn.relu),
    # keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])

    model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=2,padding='same',activation='relu',input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    ])

    batches = [5,8,16,32,64]
    for batch in batches:
        train(model,batch_size=batch,epochs=20,learning_rate=0.001)

