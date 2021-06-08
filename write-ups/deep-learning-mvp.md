# Fake Faces Generator

---

### Introduction
---

- In this project, I will apply the method of [Deep Convolutional Generative Adversarial Network - DCGAN](https://www.tensorflow.org/tutorials/generative/dcgan) to build my own low-quality fake faces generator.
- In a nutshell, the GAN model consists of 2 models (generator and discriminator) to compete with each other until some equilibrium being reached. Particularly:
    - Generator: takes a random distribution as input (typically Gaussian) and output images. 
    - Discriminator: takes either a fake image generated from the generator or a real image from the trainning set as input, and determines (binary classification task) whether the input is fake or real.
- The generator will improve upon process and gets "smarter" every epoch until the discriminator can't differentiate correctly between fake and real images, that's when we somewhat succeed.


### Training phase:
---

I train the model with **>200k** 64x64 celebrity images from this [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The training time is computationally expensive with the bendmark of **~15mins** per epoch on **Tesla P100-PCIE GPU** run in Google Colab. In order to have the decent model, it is recommended to run on ~100 epochs, which around 25hours. However, Google Colab Pro only lets the kernel running in 24hours max, so I am training my model with 95 epochs and see how it goes.

>My model is still currently running on 21/95 epochs.



### Initial findings:
---
> These are the random 10 images generated after training **GAN** model after **10** epochs.
![](https://github.com/luongtruong77/deep-learning-fake-faces-generator/blob/main/generated_images/10epochs_64x64_full.png?raw=true)

> These are the random 10 images generated after training **GAN** model after **20** epochs.
![](https://github.com/luongtruong77/deep-learning-fake-faces-generator/blob/main/generated_images/20epochs_64x64_full.png?raw=true)


### Conclusion and more work:
- We have the model to generate (somewhat) blurry and noisy images.
- They clearly don't look like real people; however, progress was made.
- Just after 20 epochs, we can see the patterns and shapes of faces.
- If time and resources permitted, we will absolutely improve our model and have higher quality images generated.
