import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
st.set_option('deprecation.showPyplotGlobalUse', False)


np.random.seed(42)

generator_30e_40k_128x128 = keras.models.load_model('models/generator_30epochs_40k_128x128.h5')
latent_dim = 128
random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
generated_images = generator_30e_40k_128x128(random_latent_vectors)

fig, ax = plt.subplots(figsize=(4, 4))

ax.imshow(generated_images[0])

plt.axis('off')


st.pyplot(fig)







# fig, ax = plt.subplots()
# im = ax.imshow(attention_array)
# st.pyplot()