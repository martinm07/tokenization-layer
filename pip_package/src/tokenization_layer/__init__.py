import tensorflow as tf # We use so much TensorFlow here we might as well just import it all.
from numpy.random import choice
from keras.initializers import Initializer
from keras.layers import Layer


def one_hot_str(string, chars):
    """
    One-hot encodes `string` using `chars`.

    Parameters
    ----------
    string : str
        String of text to one-hot encode.
    chars : str
        String of the one-hot encoding categories (i.e. characters)
        for how the text will be encoded. The index of a character
        in the string will be said character's index in the encoding.
        So, for example, `chars = "abcdefghijklmnopqrstuvwxyz"`, then
        each character in `string` will be a 26-dimensional vector 
        (i.e. a one-hot encoding with 26 characters). 
        You may also include `"<UNK>"` in your `chars` (and it'll be
        all characters that can't be found in `chars`). If you don't,
        unidentified characters will be encoded as not having a category.
        Finally, ***MAKE SURE THAT THE SAME `CHARS` ARE USED FOR ENCODING
        ALL TEXT THAT WILL BE INPUT DATA TO NEURAL NET AND INITIALIZATION
        OF PATTERNS.***

    Returns
    -------
    tf.Tensor
        `string` one-hot encoded into `len(chars)` categories.

    Examples
    --------
    ```
    >>> chars = " abcdefghijklmnopqrstuvwxyz"
    >>> # Notice how the exclamation mark doesn't have a category!
    ... tokenization_layer.one_hot_str("hello world!", chars)
    <tf.Tensor: shape=(27, 12), dtype=int32, numpy=
    array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])>
    
    >>> chars = chars + "<UNK>"
    >>> # Now with "<UNK>" it's assigned to that instead:
    ... tokenization_layer.one_hot_str("hello world!", chars)
    <tf.Tensor: shape=(28, 12), dtype=int32, numpy=
    array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])>

    >>> # Using the function on a TF Dataset
    ... example_data = tf.data.Dataset.from_tensor_slices(
    ...     tf.constant(["hello world", "This is a test", "bonjour le monde", "such is life..."]))
    ... example_data.map(lambda x: tokenization_layer.one_hot_str(x, chars))
    <MapDataset shapes: (28, None), types: tf.int32>
    ```
    """
    if chars.find("<UNK>") != -1:
        unk_index = chars.find("<UNK>")
        chars = list(chars.replace("<UNK>", ""))
        chars = chars[:unk_index] + [""] + chars[unk_index:]
        def one_hot_char(x):
            if not tf.reduce_any(x == tf.constant(chars)):
                return tf.constant([0 if i != unk_index else 1 for i in range(len(chars))])
            else:
                return tf.cast(tf.constant(list(chars))==x, tf.int32)
        return tf.transpose(tf.map_fn(one_hot_char, tf.strings.bytes_split(string), fn_output_signature=tf.int32))
    else:
        return tf.transpose(tf.map_fn(lambda x: tf.cast(tf.constant(list(chars))==x, tf.int32), 
                            tf.strings.bytes_split(string), fn_output_signature=tf.int32))


class PatternsInitilizerMaxCover(Initializer):
    """
    Keras initializer that uses a corpus of text to initialize the 
    patterns as randomly chosen grams from the corpus, weighted by 
    how common said grams are respectively.

    Parameters
    ----------
    text_corpus : str
        The corpus of text that will be used to generate the 
        patterns. WARNING: You may run into memory issues if the
        corpus is too big.
    chars : str
        String of the one-hot encoding categories (i.e. characters)
        for how the text will be encoded. The index of a character
        in the string will be said character's index in the encoding.
        So, for example, `chars = "abcdefghijklmnopqrstuvwxyz"`, then
        each character in a pattern will be 26-dimensional vector 
        (i.e. a one-hot encoding with 26 characters). 
        You may also include `"<UNK>"` in your `chars` (and it'll be
        all characters that can't be found in `chars`). If you don't,
        unidentified characters will be encoded as not having a category
        (i.e. the same as padding).
        Finally, ***MAKE SURE THAT THE SAME `CHARS` ARE USED FOR ENCODING
        ALL TEXT THAT WILL BE INPUT DATA TO NEURAL NET.***
    gram_lens : list, optional (default: `[5, 6, 7, 8, 9, 10, 11, 12, 13]`)
        A list of the possible lengths a pattern can be (patterns 
        will be padded with 0s at the end to be the same length).
    filter_over : int, optional (default: `1`)
        The minimum number of time a gram must occur in the corpus
        to be a possible pattern.

    Example
    -------
    ```
    import re
    import nltk
    nltk.download("gutenberg")
    from nltk.corpus import gutenberg

    corpus = gutenberg.raw("austen-emma.txt")
    # Remove arbritray strings of "\\n"s and " "s
    corpus = re.sub(r"[\\n ]+", " ", corpus.lower())

    chars = "".join(pd.Series(list(corpus)).value_counts(sort=True).keys()) + "<UNK>"
    init = tokenization_layer.PatternsInitilizerMaxCover(corpus, chars)

    # Initialize patterns of shape `(num_chars, max_len, 1, num_neurons)`
    #   Where there are `num_neurons` patterns (one for each neuron), each 
    #   with random length/number of characters (but padded to be `max_len`)
    #   and each character being a one-hot encoding with `num_chars` 
    #   categories.
    patterns = init((len(init.chars), max(init.gram_lens), 1, 200))
    ```
    """
    # Provide a list of possible pattern lengths (as `gram_lens`), when making a pattern it 
    # randomly chooses a length for that pattern (with uniform probability) from `gram_lens`.
    def __init__(self, text_corpus, chars, gram_lens=[5, 6, 7, 8, 9, 10, 11, 12, 13], filter_over=1):
        corpus_split = tf.strings.bytes_split(tf.constant(text_corpus))

        self.all_grams_dict, self.probs_dict = {}, {}
        self.gram_lens = gram_lens
        self.filter_over = filter_over
        self.chars = chars
        # Find the unique grams and their frequenicies for all gram_lens provided
        for gram_len in self.gram_lens:
            y, idx, count = tf.unique_with_counts(tf.strings.ngrams(corpus_split, gram_len, separator=""))

            filter_ = tf.squeeze(tf.where(count > self.filter_over))
            grams = tf.gather(y, filter_)
            filtered_count = tf.gather(count, filter_)
            
            probs = tf.expand_dims(filtered_count / tf.reduce_sum(filtered_count), 0)
            # Save the unique grams and their proabilities
            self.all_grams_dict[gram_len], self.probs_dict[gram_len] = grams, probs
    
    def __call__(self, shape, **kwargs):
        patterns = []
        # For every tokenization neuron:
        for neuron in range(shape[3]):
            # Choose a length for the pattern at random
            pattern_len = choice(self.gram_lens)
            # Choose a random gram of specified `pattern_len`
            pattern = tf.gather(self.all_grams_dict[pattern_len], tf.random.categorical(tf.math.log(self.probs_dict[pattern_len]), 1))[0, 0]
            # One-hot encode, pad, and save pattern
            pattern_onehot = tf.cast(one_hot_str(pattern, self.chars), tf.float32)
            padding = tf.zeros((pattern_onehot.shape[0], max(self.gram_lens)-pattern_len))
            patterns.append(tf.concat([pattern_onehot, padding], axis=1))
        # Concatenate patterns to make final tensor
        patterns = tf.expand_dims(tf.transpose(tf.stack(patterns), [1, 2, 0]), 2)
        # Replace 0s and 1s with random values from different distributions respectively
        return tf.where(patterns==0., tf.random.normal(patterns.shape, mean=0.25, stddev=0.08),
                                      tf.random.normal(patterns.shape, mean=0.75, stddev=0.08))
    def get_config(self):
        return {"gram_len": self.gram_lens, "filter_over": self.filter_over}

class PatternsInitializerGaussian(Initializer):
    """
    Initialize patterns using a gaussian/normal distribution.

    Parameters
    ----------
    mean : float, optional (default: `0.5`)
        Mean of gaussian/normal distribution.
    stddev : float, optional (default: `0.15`)
        Standard deviation of gaussian/normal distribution.
    
    Note that the default values (`0.5` and `0.15`) are good
    for making values between 0 and 1.
    """
    def __init__(self, mean=0.5, stddev=0.15):
        self.mean = mean
        self.stddev = stddev
    def __call__(self, shape, **kwargs):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev)


def bump_func(x, a=-0.2, b=2.5):
    a = -a/(-a+1)
    x = (1-a)*x + a
    return (x**(-b-1)) / ((1 + x**(-b) * (1 - x)**b)**2 * (-x + 1)**(-b+1))
@tf.function
def tokenization_transformation(x):
    """
    Function used to keep `patterns` within sensible range.
    """ 
    out_zeros = tf.where((x <= 0.) | (x >= 1.), 0., x)
    out_smoothstep = tf.where((out_zeros > 0) & (out_zeros < 1), bump_func(out_zeros), out_zeros)
    return out_smoothstep


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

