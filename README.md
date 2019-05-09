# Debugging Challenge

## About

The goal of this challenge is to practice debugging a neural network implementation.
This is a simple ConvNet trained on MNIST classficiation. If you fix the bugs, 
it should achieve about 99.5% accuracy on the test set after 10 epochs, which
you should be able to train on your laptop in a few minutes.

However, there are several bugs present that you will need to fix to get that 
performance. 

## Getting started

### 0. Set up a pyenv virtual environment (recommended)

Note: it is not necessary to use pyenv, but you *do need python 3.6*. 

Follow this tutorial to install pyenv and learn about it:

```
https://amaral.northwestern.edu/resources/guides/pyenv-tutorial
```

Then create a virtual env for this project:

```
pyenv virtualenv 3.6.5 debugging-challenge
pyenv activate 3.6.5/envs/debugging-challenge
```

### 1. Install requirements

Run the following:

```
pip install -r requirements.txt
```

### 2. Try to run training

Run the following:

```
python train.py
```

Instead of training on the entire dataset, you can try overfitting a single
batch by running:

```
python train.py --overfit-batch --n-epochs 200
```

### 3. Happy bug hunting! 

As a hint, there are (at least) seven bugs total in the codebase.

You can look at a corrected solution in the git branch called `working`.

<details>
  <summary>What should training look like?</summary>
  <br>
  See assets/training_example.png
</details>

<details>
  <summary>Bug 1</summary>
  <br>
  You need to pass reuse=True to the layers for the test network.
</details>
<details>
  <summary>Bug 2</summary>
  <br>
  Reshaping of the output of the pooling layer is incorrect.
</details>
<details>
  <summary>Bug 3</summary>
  <br>
  Output of the network is incorrect. The softmax cross entropy loss function 
  requires logits, but we have already taken the softmax. Change the activation
  for the last layer to None.
</details>
<details>
  <summary>Bug 4</summary>
  <br>
  Incorrect input scaling. tf.image.convert_image_dtype already scales the 
  values to [0, 1), so we are doing it twice.
</details>
<details>
  <summary>Bug 5</summary>
  <br>
  Over augmentation. Crop value of 0.1 is way too small - it's meant to be 0.9.
</details>
<details>
  <summary>Bug 6</summary>
  <br>
  Not removing augmentation at test time. augment_example method should only
  be used on the training set.
</details>
<details>
  <summary>Bug 7</summary>
  <br>
  Using regularization at test time. Dropout should be turned off at test time.
</details>