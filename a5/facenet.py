from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


def load_facenet():
    """
    loads a saved pretrained model from a json file
    :return:
    """
    # load json and create model
    json_file = open('FRmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FRmodel = model_from_json(loaded_model_json)

    # load weights into new model
    FRmodel.load_weights("FRmodel.h5")
    print("Loaded model from disk")

    return FRmodel

def img_to_encoding(img1, model):
    """
    returns 128-dimensional face embedding for input image
    :param img1:
    :param model:
    :return:
    """
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def load_dataset():

    if not os.path.exists('saved_faces/'):
        os.makedirs('saved_faces')

    a = np.load('faces.npy')

    for i in range(a.shape[0]):

        img = a[i][..., ::-1]
        img = cv2.resize(img, (96, 96))
        cv2.imwrite("saved_faces/face_image_"+str(i)+".jpg", img)

def k_clustering(model):
    imgs_des = []
    for f in os.listdir('./saved_faces'):
        embedding = img_to_encoding(cv2.imread('./saved_faces/'+f),model)[0]
        np.save('./q1_data/embedding/'+f[:-4]+'.npy',embedding)
        imgs_des.append(embedding)
    imgs_des = np.array(imgs_des)
    kmeans = KMeans(n_clusters=6).fit(imgs_des)
    return kmeans

def save_cluster_center(kmeans):
    np.save('./cluster_center.npy',kmeans.cluster_centers_)

def inverted_index(kmeans, model):
    dic = {}
    for f in os.listdir('./saved_faces'):
        classification = kmeans.predict(img_to_encoding(cv2.imread('./saved_faces/'+f),model))
        if classification[0] in dic:
            dic[classification[0]].append(f)
        else:
            dic[classification[0]] = [f]
    return dic

def find_match(kmeans,model,dic):
    df = pd.DataFrame(columns=['input_img','matching_imgs'])
    for f in os.listdir('./input_faces'):
        img = cv2.imread('./input_faces/'+f)
        img = cv2.resize(img,(96,96))
        embedding = img_to_encoding(img,model)
        classification = kmeans.predict(embedding)
        if np.linalg.norm(kmeans.cluster_centers_[classification[0]] - embedding[0]) < 0.5:
            df.loc[-1] = [f,dic[classification[0]]]
            df.index += 1
            df = df.sort_index()
        else:
            df.loc[-1] = [f,'no matches']
            df.index += 1
            df = df.sort_index()
    df.to_csv('./q2_data/matching.csv')

if __name__ == "__main__":
    load_dataset()
    fr_model = load_facenet()

    # 1c 1e save embeddings and k-means clustering
    kmeans = k_clustering(fr_model)

    # 1f save visual word and build inverted_index
    save_cluster_center(kmeans)
    dic = inverted_index(kmeans,fr_model)

    find_match(kmeans,fr_model,dic)
