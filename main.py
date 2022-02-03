import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io, color ,  util
from skimage.transform import resize
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# Image Restoration
""")

image_process = ['Colourization','Denoise','SuperResolution']
#model_name = ['CNN','RCNN','AECNN']


img_file = st.sidebar.file_uploader('Upload An Image')
sel_process = st.sidebar.selectbox("Select Restoration Process",image_process)
# sel_model = st.sidebar.selectbox("Select a model to be used",model_name)


if(sel_process == 'Colourization') :
    
    st.write(""" ### Colourization : """)
    img_size = 128

    #get input and output tensor for image:
    def ret_input_output_tensor(name,img):
        input_tensor = np.empty((1, img_size, img_size, 1))
        output_tensor = np.empty((1, img_size, img_size, 2))
        if img_file is not None :
            image = io.imread(img_file)
            image = resize(image, (img_size, img_size, 3), anti_aliasing=False, mode='constant')
            image = color.rgb2lab(image)

        if image.shape == (img_size, img_size, 3):  # if not a BW image
                # array image for output tensor
                output_tensor[0, :] = (image[:, :, 1:] / 128)
                # array values for input tensor
                input_tensor[0, :] = (image[:, :, 0] / 100).reshape(img_size, img_size, 1)

        return input_tensor, output_tensor

    #input representation
    input_tensor, output_tensor = ret_input_output_tensor(sel_process,img_file)

    # load models
    model_cnn = load_model("Colour_models/CNNs.h5")
    model_rcnn = load_model("Colour_models/RCNNs.h5")
    model_aecnn= load_model("Colour_models/AECNNs.h5")

    # predictions
    prediction_cnn = model_cnn.predict(input_tensor)
    prediction_rcnn = model_rcnn.predict(input_tensor)
    prediction_aecnn = model_aecnn.predict(input_tensor)

    #show input
    print("input tensor :\n", input_tensor[0].shape)
    input_image = np.concatenate((input_tensor[0], np.zeros((img_size, img_size, 2))), axis=2)
    input_image[:, :, 0] = input_image[:, :, 0] * 100
    input_image = color.lab2rgb(input_image)

    #show oprediction cnn 
    output_image_cnn = np.concatenate((input_tensor[0], prediction_cnn[0]), axis=2)
    output_image_cnn[:, :, 0] = output_image_cnn[:, :, 0] * 100
    output_image_cnn[:, :, 1:] = (output_image_cnn[:, :, 1:]) * 128
    output_image_cnn  = color.lab2rgb(output_image_cnn)

    #show oprediction rcnn 
    output_image_rcnn = np.concatenate((input_tensor[0], prediction_rcnn[0]), axis=2)
    output_image_rcnn[:, :, 0] = output_image_rcnn[:, :, 0] * 100
    output_image_rcnn[:, :, 1:] = (output_image_rcnn[:, :, 1:]) * 128
    output_image_rcnn  = color.lab2rgb(output_image_rcnn)

    #show oprediction aecnn 
    output_image_aecnn = np.concatenate((input_tensor[0], prediction_aecnn[0]), axis=2)
    output_image_aecnn[:, :, 0] = output_image_aecnn[:, :, 0] * 100
    output_image_aecnn[:, :, 1:] = (output_image_aecnn[:, :, 1:]) * 128
    output_image_aecnn  = color.lab2rgb(output_image_aecnn)

    #show actual
    actual_image = np.concatenate((input_tensor[0], output_tensor[0]), axis=2)
    actual_image[:, :, 0] = actual_image[:, :, 0] * 100
    actual_image[:, :, 1:] = (actual_image[:, :, 1:] ) * 128
    actual_image = color.lab2rgb(actual_image)

    #column wise output
    cols = st.columns(5)
    cols[0].image(input_image, width=220, caption="input image")
    cols[1].image(output_image_cnn ,width=220,caption="predicted_image(CNN)")
    cols[2].image(output_image_rcnn ,width=220,caption="predicted_image(RCNN)")
    cols[3].image(output_image_aecnn ,width=220,caption="predicted_image(AECNN)")
    cols[4].image(actual_image,width=220,caption="actual_image")

if(sel_process == 'Denoise') :
    
    st.write(""" ### Denoising : """)
    img_size = 128

    #get input and output tensor for image:
    def ret_input_output_tensor(name,img):
        input_tensor = np.empty((1, img_size, img_size, 3))
        output_tensor = np.empty((1, img_size, img_size, 3))
        if img_file is not None :
            image = io.imread(img_file)
            image = resize(image, (img_size, img_size, 3), anti_aliasing=False, mode='constant')
            image_n = util.random_noise(image, mode='speckle')

        if image.shape == (img_size, img_size, 3):  # if not a BW image
                # array image for output tensor
                output_tensor[0, :] = (image[:, :, :])
                # array values for input tensor
                input_tensor[0, :] = (image_n[:, :, :]).reshape(img_size, img_size, 3)

        return input_tensor, output_tensor

    #input representation
    input_tensor, output_tensor = ret_input_output_tensor(sel_process,img_file)

    # load models
    model_cnn = load_model("Denoise_models/CNNs.h5")
    model_rcnn = load_model("Denoise_models/RCNNs.h5")
    model_aecnn= load_model("Denoise_models/AECNNs.h5")

    # predictions
    prediction_cnn = model_cnn.predict(input_tensor)
    prediction_rcnn = model_rcnn.predict(input_tensor)
    prediction_aecnn = model_aecnn.predict(input_tensor)

    #show input
    print("input tensor :\n", input_tensor[0].shape)
    input_image = input_tensor[0]
    
    #show oprediction cnn 
    output_image_cnn = prediction_cnn[0]
    output_image_cnn_lab=color.rgb2lab(output_image_cnn)
    output_image_cnn_rgb=color.lab2rgb(output_image_cnn_lab)
    
    
    #show oprediction rcnn 
    output_image_rcnn = prediction_rcnn[0]
    output_image_rcnn_lab=color.rgb2lab(output_image_rcnn)
    output_image_rcnn_rgb=color.lab2rgb(output_image_rcnn_lab)

    #show oprediction aecnn 
    output_image_aecnn = prediction_aecnn[0]
    output_image_aecnn_lab=color.rgb2lab(output_image_aecnn)
    output_image_aecnn_rgb=color.lab2rgb(output_image_aecnn_lab)

    #show actual
    actual_image = output_tensor[0]

    #column wise output
    cols = st.columns(5)
    cols[0].image(input_image, width=220, caption="input image")
    cols[1].image(output_image_cnn_rgb ,width=220,caption="predicted_image(CNN)")
    cols[2].image(output_image_rcnn_rgb ,width=220,caption="predicted_image(RCNN)")
    cols[3].image(output_image_aecnn_rgb ,width=220,caption="predicted_image(AECNN)")
    cols[4].image(actual_image,width=220,caption="actual_image")

if(sel_process == 'SuperResolution') :
    
    st.write(""" ### SuperResolution : """)
    img_size = 128

    #get input and output tensor for image:
    def ret_input_output_tensor(name,img):
        input_tensor = np.empty((1, int(img_size/2), int(img_size/2), 3))
        output_tensor = np.empty((1, img_size, img_size, 3))
        if img_file is not None :
            image = io.imread(img_file)
            image = resize(image, (img_size, img_size, 3), anti_aliasing=False, mode='constant')
            image_n = resize(image, (int(img_size/2), int(img_size/2)), anti_aliasing=False, mode='constant')

        if image.shape == (img_size, img_size, 3):  # if not a BW image
                # array image for output tensor
                output_tensor[0, :] = (image[:, :, :])
                # array values for input tensor
                input_tensor[0, :] = (image_n[:, :, :]).reshape(int(img_size/2), int(img_size/2), 3)

        return input_tensor, output_tensor

    #input representation
    input_tensor, output_tensor = ret_input_output_tensor(sel_process,img_file)

    # load models
    model_cnn = load_model("SuperRe_models/CNNs.h5")
    model_rcnn = load_model("SuperRe_models/RCNNs.h5")
    model_aecnn= load_model("SuperRe_models/AECNNs.h5")

    # predictions
    prediction_cnn = model_cnn.predict(input_tensor)
    prediction_rcnn = model_rcnn.predict(input_tensor)
    prediction_aecnn = model_aecnn.predict(input_tensor)

    #show input
    print("input tensor :\n", input_tensor[0].shape)
    input_image = input_tensor[0]
    
    #show oprediction cnn 
    output_image_cnn = prediction_cnn[0]
    output_image_cnn_lab=color.rgb2lab(output_image_cnn)
    output_image_cnn_rgb=color.lab2rgb(output_image_cnn_lab)
    
    
    #show oprediction rcnn 
    output_image_rcnn = prediction_rcnn[0]
    output_image_rcnn_lab=color.rgb2lab(output_image_rcnn)
    output_image_rcnn_rgb=color.lab2rgb(output_image_rcnn_lab)

    #show oprediction aecnn 
    output_image_aecnn = prediction_aecnn[0]
    output_image_aecnn_lab=color.rgb2lab(output_image_aecnn)
    output_image_aecnn_rgb=color.lab2rgb(output_image_aecnn_lab)

    #show actual
    actual_image = output_tensor[0]

    #column wise output
    cols = st.columns(5)
    cols[0].image(input_image, width=220, caption="input image")
    cols[1].image(output_image_cnn_rgb ,width=220,caption="predicted_image(CNN)")
    cols[2].image(output_image_rcnn_rgb ,width=220,caption="predicted_image(RCNN)")
    cols[3].image(output_image_aecnn_rgb ,width=220,caption="predicted_image(AECNN)")
    cols[4].image(actual_image,width=220,caption="actual_image")


