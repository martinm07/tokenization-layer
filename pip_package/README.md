This is a package for a concept about a tokenization algorithm that is a neural network layer, training as part of a model trying to solve some NLP task, to make tokens that are best for the task. You can find more information on the [GitHub repository](https://github.com/martinm07/tokenization-layer).

#

<img src="https://imgur.com/gxxJtjz.png">

#

This package mainly consists of the `TokenizationLayer`, which is a `tf.keras` layer doing the described above. However it also contains initializers for the layer's parameter, a function for one-hot encoding a string as letters, and an Embedding layer, that doesn't have to be the first layer in the network unlike the official keras version.

Documentation for this package is [here](https://martin-github07.gitbook.io/tokenization-layer/).