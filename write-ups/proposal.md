# Fake Faces Generator
#### Can we generate fake faces using deep learning?

---


Steven L truong

---
#### What is this project about?
- Proposed in 2014 by Ian Goodfellow, [Generative Adversarial Networks (GAN)](https://arxiv.org/abs/1406.2661) has been a major player in the field of Deep Learning and Artificial Intelligence to generate new images from the train dataset.
- In a nutshell, **GAN** consists of 2 models: **generator** and **discriminator** trying to compete with each other; while generator generates new images, discriminator tests the images to see if they are real or fake. The process goes on until some equilibrium is reached.
- In this project, I would like to apply GAN to build a model to generate completely new faces (not from real people) from a collection of over 200,000 faces.

#### Task:
The task is to build the baseline model (and improved upon) to generate (probably) 128x128 new faces using GAN.

#### Data:
- The [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset with more than  **200K**  celebrity images, each with  **40**  attribute annotations. The images in this dataset cover large pose variations and background clutter. ([Multimedia Laboratory,](http://mmlab.ie.cuhk.edu.hk/) [The Chinese University of Hong Kong](http://www.cuhk.edu.hk/english/index.html))
- I intend to use small subset of the original dataset (10,000 instances) to build the baseline model and improve upon progressing.

#### Algorithm:
The main algorithm is to use **CNN** and **GAN**. Moreover, there are flavors of **GAN** I'd like to try if time and resources permitted:
- Deep Convolutional GAN (DC-GAN)
- StyleGAN (this algorithm is much more computationally expensive than GAN and DCGAN).

#### Tools:
- Python
- Pandas
- Numpy
- Tensorflow
- Keras

#### MVP:
- Some sequential models (generator and discriminator) up and running.
- Baseline model to generate *blurry* and *noisy* new faces.
    

