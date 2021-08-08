from .general import tf, Layer
from .util_functions import tokenization_transformation

@tf.custom_gradient
def tokenization(input_, patterns):
    # `upstream` will have shape (batch_size, output_height/num_neurons, output_width/input_width, 1)
    #     (you can imagine why output_height would be the same as num_neurons as "the text is now 
    #      being represented by `num_neurons` neurons, and each neuron has it's one-hot encoding, 
    #      thus, height=num_neurons". 
    #      We also specifically padded output_width to be the same as input_width in forward 
    #      propagation).
    def grad(upstream):
        # The first thing we need to do in backprop is make upstream to be as if it was
        # made using just a convolution operation (in forward prop). So, we remove the padding
        # we added in forward prop and transpose the dimensions a bit
        amount_not_padding = tf.shape(upstream)[2] - tf.shape(patterns)[1] + 1
        upstream = upstream[:, :, :amount_not_padding]
        upstream = tf.transpose(upstream, [0, 3, 2, 1])
        # Then, we need to get the gradients w.r.t. `input_` and `patterns`, pretending that forward
        # prop was just a convolution of `patterns` on `input`.
        with tf.GradientTape() as tape:
            tape.watch(input_)
            tape.watch(patterns)
            z = tf.nn.convolution(input_, patterns) * upstream # Note how we multiply (element-wise)
            # with `upstream`!
        grads = tape.gradient(z, [input_, patterns])
        # Finally, we return the gradients (remembering to multiply the patterns' gradients 
        # element-wise with sigma(x)).
        return grads[0], grads[1]*tokenization_transformation(patterns)
    
    # The `patterns` parameter is just arbritrary numbers. We want `patterns` to be able to match
    # the text. For that each column must contain one "1" and the rest be zeros
    # (one-hot encoding), to make that we take the maximum value in each column and say that's
    # the "1". We would also like there to be columns that are ignored. Those columns would be
    # filled with just zeros. To do that, we say that if the sum of all the values in a column
    # is less than (or equal to) 0, that column if filled with just zero.
    patterns_discrete = tf.cast(tf.math.logical_and(
        patterns == tf.expand_dims(tf.reduce_max(patterns, axis=0), 0),
        tf.reduce_sum(patterns, axis=0) > 0
    ), tf.float32)
    # Convolve the input with patterns_discrete
    convolution = tf.nn.convolution(input_, patterns_discrete)
    # Pad convolution to be the same length as `input_`
    padding = tf.zeros([tf.shape(convolution)[0], tf.shape(convolution)[1], 
                        tf.shape(input_)[2]-tf.shape(convolution)[2], tf.shape(convolution)[3]])
    convolution = tf.concat([convolution, padding], axis=2)
    # Get the sums of all the different patterns respectively (this is so we
    #  can find where in the text/s the pattern/s made a full match).
    pattern_sums = tf.reduce_sum(patterns_discrete, axis=[0, 1])

    # This function will take in one instance in the batch at a time
    def transform_convolution(t):
        return tf.transpose(tf.cast(tf.squeeze(t) == pattern_sums, tf.float32))
    # Map `transform_convolution` over every instance in the batch
    final = tf.map_fn(transform_convolution, convolution)
    # Append an extra "1" in the shape (to be more like the input) and return
    return tf.expand_dims(final, 3), grad

class TokenizationLayer(Layer):
    """
    Placed as the first layer in a neural network, it takes in text that's
    split by character and one-hot encoded (i.e. has shape 
    `(batch_size, num_chars, text_len, 1)`), and "tokenizes" it using the
    trainable parameter `patterns`. CURRENTLY IT DOES NOT PERFORM WELL AND
    SHOULDN'T BE USED IN ANY REAL TASKS.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to be in the layer.
    initializer : Initializer
        Initializer for patterns.
    pattern_lens : int
        Length/Number of characters every pattern will be.
    **kwargs

    Example
    -------
    ```
    from tensorflow import keras
    import re
    import nltk
    nltk.download("gutenberg")
    from nltk.corpus import gutenberg

    corpus = gutenberg.raw("austen-emma.txt")
    # Remove arbritray strings of "\\n"s and " "s
    corpus = re.sub(r"[\\n ]+", " ", corpus.lower())

    # We're assuming we got `chars` when preprocessing the train data
    init = tokenization_layer.PatternsInitilizerMaxCover(corpus, chars)

    model = keras.Sequential([
        tokenization_layer.TokenizationLayer(500, init, max(init.gram_lens)),
        Lambda(lambda x: tf.transpose(tf.squeeze(x, 3), [0, 2, 1])),
        tokenization_layer.EmbeddingLayer(1),
        Flatten(),
        BatchNormalization(),
        Dense(64),
        Dense(1, activation="sigmoid")
    ])
    # Initialize parameters and shapes by calling on dummy inputs
    _ = model(tf.zeros((32, 30, 2000, 1)))
    _ = model(tf.zeros((50, 30, 2000, 1)))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_data, epochs=10)
    ```
    """
    def __init__(self, n_neurons, initializer, pattern_lens, **kwargs):
        super().__init__(**kwargs)

        self.n_neurons = n_neurons
        self.initializer = initializer
        self.pattern_lens = pattern_lens
    
    def build(self, input_shape):
        # Initialize parameters of TokenizationLayer
        self.patterns = self.add_weight("patterns", shape=[input_shape[1], self.pattern_lens, 1, self.n_neurons],
                                        initializer=self.initializer)
        super().build(input_shape)

    def call(self, input_):
        return tokenization(input_, self.patterns)
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[0]+[self.n_neurons]+input_shape.as_list()[2]+[1])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "n_neurons":self.n_neurons, "initializer":self.initializer, "pattern_lens":self.pattern_lens}

class EmbeddingLayer(Layer):
    """
    Keras layer that takes in discrete values/categories (one-hot encoded)
    and embeds (maps) them to continuous values. This embedding is learnt 
    through training. Shape of input should be 
    `(batch_size, sequence_length, onehot_categories)`.
    Note that this is essentially the same as `Embedding`,
    except that this version doesn't have to be the first layer in the
    network (i.e. it has an upstream gradient), unlike the official
    keras embedding layer.
    
    Parameters
    ----------
    embedding_length : int
        How many values (i.e. dimensions in the vector) each category should
        be embedded as. The more values the more complex the *meaning* behind
        any embedding can be.
    **kwargs
    """
    def __init__(self, embedding_length, **kwargs):
        super().__init__(**kwargs)
        self.embedding_length = embedding_length
    
    # input_shape = (batch_size, sequence_length, num_unique_vals)
    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(name="embedding_matrix", 
                                                shape=[input_shape[2], self.embedding_length],
                                                initializer="glorot_uniform")
        super().build(input_shape)
    
    def call(self, X):
        return tf.matmul(X, self.embedding_matrix)
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[:-1] + [self.embedding_length])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "embedding_length": self.embedding_length}