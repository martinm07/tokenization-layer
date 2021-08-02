# Tokenization Layer
[![View in Deepnote](https://deepnote.com/static/buttons/view-in-deepnote-white.svg)](https://deepnote.com/viewer/github/martinm07/tokenization-layer/blob/main/concept-explained.ipynb)

This is a concept for a tokenization algorithm that is a neural network layer, training as part of a model trying to solve some NLP task, to make tokens that are best for the task. This is explained further in `concept_explained.ipynb`.

#

<img src="https://imgur.com/gxxJtjz.png">

#

The tokenization layer is a layer that takes in text, split by letter and one-hot encoded, and outputs the same text, but represented by it's *patterns* (i.e. tokens) instead of letters. In other words, the tokenization layer returns a text of 0s, except with the patterns' *signatures* ("out") wherever said patterns were in the text. **Here's an example:**

<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Ctext%7B%22a%22%7D%3D%5Cbegin%7Bbmatrix%7D1%5C%5C0%5C%5C0%5C%5C0%5C%5C0%5Cend%7Bbmatrix%7D%20%5Ctext%7B%22b%22%7D%3D%5Cbegin%7Bbmatrix%7D0%5C%5C1%5C%5C0%5C%5C0%5C%5C0%5Cend%7Bbmatrix%7D%20%5Ctext%7B%22c%22%7D%3D%5Cbegin%7Bbmatrix%7D0%5C%5C0%5C%5C1%5C%5C0%5C%5C0%5Cend%7Bbmatrix%7D%20%5Ctext%7B%22d%22%7D%3D%5Cbegin%7Bbmatrix%7D0%5C%5C0%5C%5C0%5C%5C1%5C%5C0%5Cend%7Bbmatrix%7D%20%5Ctext%7B%22%20%22%7D%3D%5Cbegin%7Bbmatrix%7D0%5C%5C0%5C%5C0%5C%5C0%5C%5C1%5Cend%7Bbmatrix%7D"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Ctext%7Btext%20%3D%20%22a%20bad%20cab%22%7D%20%5Clongmapsto%20%5Cbegin%7Bbmatrix%7D%201%260%260%261%260%260%260%261%260%20%5C%5C%200%260%261%260%260%260%260%260%261%20%5C%5C%200%260%260%260%260%260%261%260%260%20%5C%5C%200%260%260%260%261%260%260%260%260%20%5C%5C%200%261%260%260%260%261%260%260%260%20%5Cend%7Bbmatrix%7D"></p>

<p align="center">^ <i>Showing how the tokenization layer takes in text that is split by letter and one-hot encoded</i></p>

<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Ctext%7Bpattern%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%261%260%20%5C%5C%201%260%260%20%5C%5C%200%260%260%20%5C%5C%200%260%261%20%5C%5C%200%260%260%20%5Cend%7Bbmatrix%7D%20%5Ctext%7B%2C%20out%7D%20%3D%5Cbegin%7Bbmatrix%7D1%20%5C%5C0%20%5C%5C0%20%5C%5C0%20%5C%5C0%5Cend%7Bbmatrix%7D"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?f%28text%2C%20pattern%2C%20out%29%20%3D%20%5Ctext%7Breturn%20%7Dtext%5Ctext%7B%20of%200s%20except%20with%20%7Dout%5Ctext%7B%20where%20%7Dpattern%5Ctext%7B%20in%20%7Dtext"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?f%28%5Ctext%7Btext%7D%2C%20%5Ctext%7Bpattern%7D%2C%20%5Ctext%7Bout%7D%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"></p>

<p align="center">^ <i>Showing how a single neuron of the tokenization layer works</i></p>

<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20f_1%28t%2C%20p_1%2C%20o_1%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D%2C%20f_2%28t%2C%20p_2%2C%20o_2%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20%2C%20f_3%28t%2C%20p_3%2C%20o_3%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%201%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D%2C%20f_4%28t%2C%20p_4%2C%20o_4%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%201%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20%2C%20f_5%28t%2C%20p_5%2C%20o_5%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Ctext%7Blayer%20output%7D%20%3D%20f_1%28t%2C%20p_1%2C%20o_1%29%20&plus;%20f_2%28t%2C%20p_2%2C%20o_2%29%20&plus;%20f_3%28t%2C%20p_3%2C%20o_3%29%20&plus;%20f_4%28t%2C%20p_4%2C%20o_4%29%20&plus;%20f_5%28t%2C%20p_5%2C%20o_5%29%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%201%20%26%200%20%5C%5C%200%20%26%200%20%26%200%20%26%200%20%26%201%20%26%200%20%26%200%20%26%200%20%26%201%20%5C%5C%201%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"></p>

<p align="center">^ <i>Showing how the multiple neurons in a tokenization layer come together to make the layer output</i></p>

These patterns (i.e. tokens) should then update through training, and to do that we need to get the derivative of the layer w.r.t. the patterns. To do that we rewrite our layer to use convolutions (from CNNs), and derive from there. With this (and a couple finer details) we have the *tokenization layer*.

However, as it stands now, it is unable to really train. At a high level, this is due to the fact that there are many possible tokens that will never be detected in a text (i.e. random strings of characters), and that the upstream gradient at the layer is probably impossible for it to follow (as in, the rest of the neural network wants the layer to produce outputs that are impossible for it to output).<br>
Again, for more details refer to `concept_explained.ipynb`.

***

If you would like a stable environment to work on this repository, there's [Deepnote](https://deepnote.com/home), which is where most of the work for this repository was done. Just create a new project and integrate with this repository.

Despite that, here's the `requirements.txt`:<br>
```
colorama==0.4.4
seaborn==0.11.1
matplotlib==3.3.4
tensorflow==2.5.0
pandas==1.2.4
numpy==1.19.5
nltk==3.6.1
ipython==7.26.0
scikit_learn==0.24.2
```

