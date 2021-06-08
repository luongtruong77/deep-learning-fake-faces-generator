import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.set_option('deprecation.showPyplotGlobalUse', False)

########################################

st.markdown("<h1 style='text-align: center; color: black;'>Fake Faces Generator!</h1>",
            unsafe_allow_html=True)
st.write('---')

########################################

st.sidebar.write('**Home Page**')
st.sidebar.write('---')
st.sidebar.write('This fun generator is built using **Deep Convolutional Generative Adversarial Network** [**(DCGAN)**]'
                 '(https://arxiv.org/abs/1406.2661).'
                 ' The codes can be found [**HERE**]'
                 '(https://github.com/luongtruong77/deep-learning-fake-faces-generator).')
st.sidebar.write('The model was trained with over **200k** 64x64 images from the [CelebA dataset]'
                 '(http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) on **95** epochs using **Tesla P100-PCIE** '
                 'GPU on Google Colab in ~23hours. The quality of generated images will typically improve upon'
                 'increasing number of epochs (try 200 for instance).')

st.write('### **What are GANs?**')
st.write('[Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661) were proposed in 2014 by '
         '[Ian Goodfellow](https://www.linkedin.com/in/ian-goodfellow-b7187213/) and they have been major players '
         'in the field of machine learning and deep learning. Two models are trained simultaneously by an adversarial '
         'process. A generator ("the artist") learns to create images that look real, while a discriminator '
         '("the art critic") learns to tell real images apart from fakes.')
st.write('- **Generator:** takes a random distribution as input (typically Gaussian) and outputs images.')
st.write('- **Discriminator:** takes either a fake image generated from the generator or a real image from the '
         'training set as input, and determines (binary classification task) whether the input is fake or real.')
st.write("The generator will improve upon process and gets 'smarter' every epoch until the discriminator can't "
         "differentiate correctly between fake and real images, that's when we somewhat succeed.")
st.write("Let's generate some fake faces!")
st.write('---')
st.write('**Note:** The generator typically takes around 1-5 sec to load depends on your device, your internet '
         'connection and your browser.')
if st.button('GENERATE'):
    generator_30e_40k_128x128 = keras.models.load_model('models/generator_30epochs_40k_128x128.h5')
    latent_dim = 128
    random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
    generated_images = generator_30e_40k_128x128(random_latent_vectors)

    fig, ax = plt.subplots(figsize=(10, 10))

    st.image(keras.preprocessing.image.array_to_img(generated_images[0]), width=200)
    plt.axis('off')

    st.pyplot(fig)


