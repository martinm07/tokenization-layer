from .general import tf

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