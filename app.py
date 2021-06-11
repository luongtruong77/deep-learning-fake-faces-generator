import streamlit as st
import matplotlib.pyplot as plt
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
                 '(http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) on different number of epochs number of training '
                 'instances using **Tesla P100** '
                 'GPU on Google Cloud in ~24hours. The quality of generated images will typically improve upon'
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

#############################################

st.write('**Note:** The generator typically takes around 1-5 sec to load depends on your device, your internet '
         'connection and your browser.')
st.write('Understanding the generating buttons:')
st.write('- 150e: 150 epochs were run on.')
st.write('- 100k: 100,000 training data points (images) were trained on.')
st.write('- 64x64: Resolution of training images were trained on.')
choices = st.radio('Please choose how you want to generate images:', ('10 images at a time', '1 image at a time'))

latent_dim = 128

generator_150e_100k_64x64 = keras.models.load_model('models/generator_150epochs_100k_64x64.h5')
generator_60e_200k_64x64 = keras.models.load_model('models/generator_60epochs_64x64.h5')
list_images = []
if choices == '10 images at a time':


    if st.button('GENERATE (150e_100k_64x64)'):
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = generator_150e_100k_64x64(random_latent_vectors)

        for i in range(10):
            list_images.append(generated_images[i])

        # if len(list_images) >= 30:
        #     list_images = list_images[10:]
        # else:
        #     pass
        st.write(len(list_images))
        plt.figure(figsize=(12, 5 * (len(list_images) // 10)))
        for i in range(len(list_images)):
            plt.subplot(len(list_images) // 5, 5, i + 1)
            plt.imshow(list_images[i])
            plt.axis('off')

        st.pyplot()

    if st.button('GENERATE (60e_200k_64x64)'):
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = generator_60e_200k_64x64(random_latent_vectors)
        plt.figure(figsize=(14, 5))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(generated_images[i])
            plt.axis('off')
        st.pyplot()

    if st.button('GENERATE (100e_200k_64x64'):
        st.write('Coming soon...')
        st.write('I am working on the next generator with 100 epochs and 200,000 training images.')

elif choices == '1 image at a time':

    if st.button('GENERATE (150e_100k_64x64)'):
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = generator_150e_100k_64x64(random_latent_vectors)
        fig, ax = plt.subplots(figsize=(10, 10))
        st.image(keras.preprocessing.image.array_to_img(generated_images[0]), width=200)
        plt.axis('off')
        st.pyplot(fig)

    if st.button('GENERATE (60e_200k_64x64)'):
        random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
        generated_images = generator_60e_200k_64x64(random_latent_vectors)
        fig, ax = plt.subplots(figsize=(10, 10))
        st.image(keras.preprocessing.image.array_to_img(generated_images[0]), width=200)
        plt.axis('off')
        st.pyplot(fig)

    if st.button('GENERATE (100e_200k_64x64'):
        st.write('Coming soon...')
        st.write('I am working on the next generator with 100 epochs and 200,000 training images.')
