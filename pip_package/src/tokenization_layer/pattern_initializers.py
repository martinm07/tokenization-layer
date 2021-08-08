from .general import tf, choice, Initializer
from .util_functions import one_hot_str

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