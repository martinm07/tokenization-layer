# THIS SCRIPT TRAINS A MODEL (USING THE TOKENIZATION LAYER) ON THE IMDB MOVIE REVIEWS DATASET
path = input("File path to make model checkpoints and save training statistics (should be empty): ")
epochs = int(input("Number of epochs to train for (default 10): ") or "10")
# ---------------------------------------

# General libraries
import numpy as np
import pandas as pd

# ML & DL libraries
import tensorflow as tf
from tensorflow import keras
print(f"TensorFlow version: {tf.__version__}")

# Misc libraries
import textwrap
import re
import string
from copy import copy
import time
import os
from os import system, name
from colorama import Fore

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# define a function to clear output
def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

# Import and prepare data
data = pd.read_csv("example_datasets/IMDB Dataset.csv")
data = data.sample(len(data)).reset_index(drop=True)

# Strip "<br />" tags and convert to lowercase
data["review"] = data["review"].apply(lambda x: x.replace("<br />", " ").lower())
# Strip punctuation
data["review"] = data["review"].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))
# Get top 30 most common characters
chars = "".join(pd.Series(list(" ".join(data["review"].to_list()))).value_counts().keys()[:30])
# Remove everything except the top 30 most common characters
data["review"] = data["review"].apply(lambda x: re.sub(f"[^{chars}]", "", x))

from sklearn.preprocessing import OrdinalEncoder
data[["sentiment"]] = OrdinalEncoder().fit_transform(data[["sentiment"]])

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data["review"], data["sentiment"], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

@tf.function
# Encode characters into integers
def ordinal_encode(x):
    split = tf.strings.bytes_split(x)
    chars_tensor = tf.constant(list(chars))
    return tf.argmax(tf.map_fn(lambda x: chars_tensor==x, split, dtype=tf.bool), axis=1)

# One-hot encode integers
def onehot(x):
    return tf.transpose(tf.one_hot(x, depth=len(chars)))

# Clip and pad all reviews to be 2000 characters long
def clip_and_pad(x):
    output_length = 2000
    shape = tf.shape(x)
    if shape[1] >= output_length:
        return x[:, :output_length]
    else:
        return tf.concat([x, tf.zeros((shape[0], output_length-shape[1]))], axis=1)

X_train, X_val, X_test = tf.data.Dataset.from_tensor_slices(X_train), tf.data.Dataset.from_tensor_slices(X_val), tf.data.Dataset.from_tensor_slices(X_test)
X_train, X_val, X_test = X_train.map(ordinal_encode).map(onehot).map(clip_and_pad), X_val.map(ordinal_encode).map(onehot).map(clip_and_pad), X_test.map(ordinal_encode).map(onehot).map(clip_and_pad)

y_train, y_val, y_test = tf.data.Dataset.from_tensor_slices(np.asarray(y_train).astype('float32')), tf.data.Dataset.from_tensor_slices(np.asarray(y_val).astype('float32')), tf.data.Dataset.from_tensor_slices(np.asarray(y_test).astype('float32'))

train_set, val_set, test_set = tf.data.Dataset.zip((X_train, y_train)), tf.data.Dataset.zip((X_val, y_val)), tf.data.Dataset.zip((X_test, y_test))
for item in train_set.take(3):
    print(item)

train_set = train_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)
val_set = val_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)
test_set = test_set.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False) \
                     .batch(32, drop_remainder=True).prefetch(1)

train_set = train_set.map(lambda x, y: (tf.expand_dims(x, 3), y))
val_set = val_set.map(lambda x, y: (tf.expand_dims(x, 3), y))
test_set = test_set.map(lambda x, y: (tf.expand_dims(x, 3), y))

char_lookup = tf.concat([tf.constant(["█"]), tf.strings.bytes_split(tf.constant(chars))], axis=0)
reverse_text = lambda x: tf.strings.join(tf.gather(char_lookup, tf.argmax(tf.concat([tf.fill([1, x.shape[1]], 0.5), x], axis=0), axis=0)))

# Create intialization for patterns, `PatternsInitilizerMaxCover`
def gram_count_batch(gram_len=10, filter_over=1):
    gram_text = lambda x: tf.strings.ngrams(tf.strings.bytes_split(x), gram_len, separator="")
    def return_func(x):
        grammed_batch = tf.ragged.map_flat_values(gram_text, x)
        flattened_batch = grammed_batch.merge_dims(0, -1)
        y, idx, count = tf.unique_with_counts(flattened_batch)
        filter_ = tf.squeeze(tf.where(count > filter_over), 1)
        filtered_y = tf.gather(y, filter_)
        filtered_count = tf.gather(count, filter_)
        return (filtered_y, filtered_count)
    return return_func

def one_hot_str(string):
    return tf.transpose(tf.map_fn(lambda x: tf.cast(tf.constant(list(chars))==x, tf.int32), 
                        tf.strings.bytes_split(string), fn_output_signature=tf.int32))

X_train, X_val, y_train, y_val = train_test_split(data["review"], data["sentiment"], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
X_train_data = tf.data.Dataset.from_tensor_slices(X_train)
X_train_batched = X_train_data.batch(1000)

class PatternsInitilizerMaxCover(keras.initializers.Initializer):
    # Provide a list of possible pattern lengths (as `gram_lens`), when making a pattern it 
    # randomly chooses a length for that pattern (with uniform probability) from `gram_lens`.
    def __init__(self, gram_lens, filter_over=1):
        self.all_grams_dict, self.probs_dict = {}, {}
        self.gram_lens = gram_lens
        # Find the unique grams and their frequenicies for all gram_lens provided
        for gram_len in self.gram_lens:
            # Get all unique grams and their counts (in batches, because all at once eats the RAM)
            X_train_batch_counted = X_train_batched.map(gram_count_batch(gram_len=gram_len, filter_over=filter_over))
            gram_counts_batched = []
            for batch in X_train_batch_counted:
                gram_counts_batched.append(batch)
            df = pd.DataFrame(gram_counts_batched)
            # Concatenate batches together
            all_counts = tf.concat(df[1].to_list(), axis=0)
            all_grams = tf.concat(df[0].to_list(), axis=0)
            # Normalize the counts to get the "probabilites"
            probs = tf.expand_dims(all_counts / tf.reduce_sum(all_counts), 0)
            # Save the unique grams and their proabilities
            self.all_grams_dict[gram_len], self.probs_dict[gram_len] = all_grams, probs
    
    def __call__(self, shape, **kwargs):
        patterns = []
        # For every tokenization neuron:
        for neuron in range(shape[3]):
            # Choose a length for the pattern at random
            pattern_len = np.random.choice(self.gram_lens)
            # Choose a random gram of specified `pattern_len`
            pattern = tf.gather(self.all_grams_dict[pattern_len], tf.random.categorical(tf.math.log(self.probs_dict[pattern_len]), 1))[0, 0]
            # One-hot encode, pad, and save pattern
            pattern_onehot = tf.cast(one_hot_str(pattern), tf.float32)
            padding = tf.zeros((pattern_onehot.shape[0], max(self.gram_lens)-pattern_len))
            patterns.append(tf.concat([pattern_onehot, padding], axis=1))
        # Concatenate patterns to make final tensor
        patterns = tf.expand_dims(tf.transpose(tf.stack(patterns), [1, 2, 0]), 2)
        # Replace 0s and 1s with random values from differnet distributions respectively
        return tf.where(patterns==0., tf.random.normal(patterns.shape, mean=0.25, stddev=0.08),
                                      tf.random.normal(patterns.shape, mean=0.75, stddev=0.08))
    def get_config(self):
        return {"gram_len": gram_len, "filter_over": filter_over}

# Define sigma(x)
def bump_func(x, a=-0.2, b=2.5):
    a = -a/(-a+1)
    x = (1-a)*x + a
    return (x**(-b-1)) / ((1 + x**(-b) * (1 - x)**b)**2 * (-x + 1)**(-b+1))
@tf.function
def tokenization_transformation(x):
    out_zeros = tf.where((x <= 0.) | (x >= 1.), 0., x)
    out_smoothstep = tf.where((out_zeros > 0) & (out_zeros < 1), bump_func(out_zeros), out_zeros)
    return out_smoothstep

# Define tokenization layer
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

class TokenizationLayer(keras.layers.Layer):
    def __init__(self, n_neurons, possible_pattern_lens=[5, 6, 7, 8, 9, 10, 11, 12, 13], filter_over=1, **kwargs):
        super().__init__(**kwargs)

        self.n_neurons = n_neurons
        self.possible_pattern_lens = possible_pattern_lens
        self.filter_over = filter_over
    
    def build(self, input_shape):
        # Initialize parameters of TokenizationLayer
        self.patterns = self.add_weight("patterns", shape=[input_shape[1], max(self.possible_pattern_lens), 1, self.n_neurons],
                                        initializer=PatternsInitilizerMaxCover(self.possible_pattern_lens, filter_over=self.filter_over))
        super().build(input_shape)

    def call(self, input_):
        return tokenization(input_, self.patterns)
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape.as_list()[0]+[self.n_neurons]+input_shape.as_list()[2]+[1])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "n_neurons":self.n_neurons, "possible_pattern_lens":self.possible_pattern_lens,
                "filter_over":self.filter_over}

# Define Embedding layer
class MyEmbedding(keras.layers.Layer):
    """
    Takes in matrix of discrete values (one-hot encoded) and embeds them into 
    continuous values, trained like the rest of the network.
    Shape of `X` should be `(batch_size, sequence_length, onehot_categories)`
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

# Define model
class ModelTokenization(tf.keras.Model):
    def __init__(self):
        super(ModelTokenization, self).__init__(name='')
        
        self.tokenization = TokenizationLayer(n_neurons=500)
        self.lambda1 = keras.layers.Lambda(lambda x: tf.transpose(tf.squeeze(x, 3), [0, 2, 1]))
        self.embedding = MyEmbedding(1)
        self.flatten = keras.layers.Flatten()

        self.batch_norm1 = keras.layers.BatchNormalization()
        self.dense = keras.layers.Dense(64)
        self.out = keras.layers.Dense(1, activation="sigmoid")

    def call(self, input_tensor, return_intermediates=False, training=False):
        tokenization_out = self.tokenization(input_tensor, training=training)
        lambda1_out = self.lambda1(tokenization_out, training=training)
        embedding_out = self.embedding(lambda1_out, training=training)
        flatten_out = self.flatten(embedding_out)

        batch_norm1_out = self.batch_norm1(flatten_out, training=training)
        dense_out = self.dense(batch_norm1_out, training=training)
        out = self.out(dense_out, training=training)

        if return_intermediates:
            return out, dense_out, flatten_out, embedding_out, lambda1_out, tokenization_out
        else:
            return out

# Initialize model
model = ModelTokenization()
_ = model(tf.zeros([32, 30, 2000, 1]))

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy()
train_acc_metric = keras.metrics.Accuracy()
val_acc_metric = keras.metrics.Accuracy()

with open(path+"patterns_log.txt", "a+") as f:
    pass
with open(path+"grads_log.csv", "a+") as f:
    f.write("Out Mean,Out Std,Dense Mean,Dense Std,Embedding Mean,Embedding Std,Tokenization Mean,"+\
            "Tokenization Std,Out Kernel Mean,Out Kernel Std,Out Bias Mean,Out Bias Std,"+\
            "Dense Kernel Mean,Dense Kernel Std,Dense Bias Mean,Dense Bias Std,"+\
            "Embedding Kernel Mean,Embedding Kernel Std,Patterns Mean,Patterns Std,\n")
with open(path+"vals_log.csv", "a+") as f:
    f.write("Out Mean,Out Std,Dense Mean,Dense Std,Embedding Mean,Embedding Std,Tokenization Mean,"+\
            "Tokenization Std,Out Kernel Mean,Out Kernel Std,Out Bias Mean,Out Bias Std,"+\
            "Dense Kernel Mean,Dense Kernel Std,Dense Bias Mean,Dense Bias Std,"+\
            "Embedding Kernel Mean,Embedding Kernel Std,Patterns Mean,Patterns Std,\n")

# Train model
for epoch in range(epochs):
    train_loss_rounded, train_acc_rounded = 0, 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_set):
        # -=-= COMPUTE GRADIENTS OF BATCH  =-=-
        with tf.GradientTape() as tape:
            z, dense_out, flatten_out, embedding_out, lambda1_out, tokenization_out = model(x_batch_train, return_intermediates=True, training=True)
            z = tf.squeeze(z, 1)
            loss = loss_fn(y_batch_train, z)
        layer_vals = [z, dense_out, embedding_out, tokenization_out]
        grads = tape.gradient(loss, layer_vals+model.trainable_variables)
        layer_grads = grads[:len(layer_vals)]
        grads = grads[len(layer_vals):]
        
        # -=-= LOG INGO  =-=-
        progress_bar_done = "".join(["█" for _ in range(round( step*20/len(train_set) ))])
        progress_bar_left = "".join([" " for _ in range(20-round( step*20/len(train_set) ))])
        percent_done = round(step*100/len(train_set), 2)

        save_patterns = False
        if step%10 == 0:
            save_patterns = True
            # Decode patterns
            patterns = model.tokenization.patterns
            patterns = tf.cast(tf.math.logical_and(
                patterns == tf.expand_dims(tf.reduce_max(patterns, axis=0), 0),
                tf.reduce_sum(patterns, axis=0) > 0
            ), tf.float32)
            patterns_decoded = [reverse_text(pattern).numpy().decode() for pattern in tf.transpose(tf.squeeze(patterns, 2), [2, 0, 1])]
            # Get patterns to log
            pattern_grads = tf.transpose(tf.squeeze(grads[0], 2), [2, 0, 1])
            pattern_grads_summary = tf.math.reduce_std(pattern_grads, axis=[1, 2])+tf.abs(tf.reduce_mean(pattern_grads, axis=[1, 2]))
            pattern_grads_sorted_indexes = list(pd.Series(pattern_grads_summary).sort_values().keys())

        clear()
        
        print(f'Epoch {epoch+1}/{epochs} - |{progress_bar_done}{progress_bar_left}| - {percent_done}% - {step+1}/{len(train_set)}')
        print(f'Train loss: {train_loss_rounded} - Train accuracy: {train_acc_rounded}')
        print()
        # Log patterns
        top_n = 15
        buffer = "".join("0" for _ in range(7))

        patterns_log_high = [f'"{patterns_decoded[i]}": '+(str(pattern_grads_summary[i].numpy()*100)+buffer)[:7]+" | " 
                             for i in pattern_grads_sorted_indexes[-top_n:]]
        num_per_row = int(np.floor(135/len(patterns_log_high[0])))
        print(f"{color.BOLD}Patterns with diverse non-zero gradients{color.END}")
        for i in range(int(np.floor(len(patterns_log_high)/num_per_row))):
            print("".join(patterns_log_high[(i)*num_per_row:(i+1)*num_per_row]))
        if len(patterns_log_high)%num_per_row != 0:
            print("".join(patterns_log_high[-(int(np.floor(len(patterns_log_high)/num_per_row))*num_per_row)+1:]))

        patterns_log_low = [f'"{patterns_decoded[i]}": '+(str(pattern_grads_summary[i].numpy()*100)+buffer)[:7]+" | " 
                             for i in pattern_grads_sorted_indexes[:top_n]]
        num_per_row = int(np.floor(135/len(patterns_log_low[0])))
        print(f"{color.BOLD}Patterns with mostly zero gradients{color.END}")
        for i in range(int(np.floor(len(patterns_log_low)/num_per_row))):
            print("".join(patterns_log_low[(i)*num_per_row:(i+1)*num_per_row]))
        if len(patterns_log_low)%num_per_row != 0:
            print("".join(patterns_log_low[-(int(np.floor(len(patterns_log_low)/num_per_row))*num_per_row)+1:]))

        # -=-= UPDATE NETWORK & METRICS  =-=-
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(y_batch_train, tf.round(z))
        train_loss_rounded, train_acc_rounded = "%.4f" % loss.numpy(), "%.4f" % train_acc_metric.result().numpy()

        # -=-= SAVE THINGS TO DRIVE  =-=-
        if (step%int(np.floor(len(train_set)/5))==0) and (step != 0):
            cp_num = len(os.listdir(path))-1
            model_cp = tf.train.Checkpoint(model=model)
            model_cp.write(path+f"model_cp_{cp_num}/model_checkpoint")
        if save_patterns:
            with open(path+"patterns_log.txt", "a+") as f:
                [f.write(f'"{pattern}", ') for pattern in patterns_decoded]
                f.write("\n")
        with open(path+"grads_log.csv", "a+") as f:
            for layer_index in [0, 1, 2, 3]:
                f.write(str( tf.reduce_mean(layer_grads[layer_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(tf.reduce_mean(layer_grads[layer_index], axis=0)).numpy() )+",")
            for param_index in [6, 7, 4, 5, 1, 0]:
                f.write(str( tf.reduce_mean(grads[param_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(grads[param_index]).numpy() )+",")
            f.write("\n")
        with open(path+"vals_log.csv", "a+") as f:
            for layer_index in [0, 1, 2, 3]:
                f.write(str( tf.reduce_mean(layer_vals[layer_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(tf.reduce_mean(layer_vals[layer_index], axis=0)).numpy() )+",")
            for param_index in [6, 7, 4, 5, 1, 0]:
                f.write(str( tf.reduce_mean(model.trainable_variables[param_index]).numpy() )+",")
                f.write(str( tf.math.reduce_std(model.trainable_variables[param_index]).numpy() )+",")
            f.write("\n")

    train_acc_metric.reset_states()

