# Tokenization Layer
[![View in Deepnote](https://deepnote.com/static/buttons/view-in-deepnote-white.svg)](https://deepnote.com/viewer/github/martinm07/tokenization-layer/blob/main/concept-explained.ipynb)

This is a concept for a tokenization algorithm that is a neural network layer, training as part of a model trying to solve some NLP task, to make tokens that are best for the task. This is explained further in `concept_explained.ipynb`.

The tokenization layer is a layer that takes in text, split by letter and one-hot encoded, and outputs the same text, but represented by it's *patterns* (i.e. tokens) instead of letters. In other words, the tokenization layer returns a text of 0s, except with the patterns' *signatures* ("out") wherever said patterns were in the text. **Here's an example:**

$$
\text{"a"}=\begin{bmatrix}1\\0\\0\\0\\0\end{bmatrix} \text{"b"}=\begin{bmatrix}0\\1\\0\\0\\0\end{bmatrix} \text{"c"}=\begin{bmatrix}0\\0\\1\\0\\0\end{bmatrix} \text{"d"}=\begin{bmatrix}0\\0\\0\\1\\0\end{bmatrix} 
\text{" "}=\begin{bmatrix}0\\0\\0\\0\\1\end{bmatrix}
$$

$$
\text{text = "a bad cab"} \longmapsto \begin{bmatrix}
1&0&0&1&0&0&0&1&0 \\
0&0&1&0&0&0&0&0&1 \\
0&0&0&0&0&0&1&0&0 \\
0&0&0&0&1&0&0&0&0 \\
0&1&0&0&0&1&0&0&0
\end{bmatrix}
$$

<p align="center"><i>Showing how the tokenization layer takes in text that is split by letter and one-hot encoded</i></p>

$$
\text{pattern} =
\begin{bmatrix}
0&1&0 \\
1&0&0 \\
0&0&0 \\
0&0&1 \\
0&0&0
\end{bmatrix}
\text{, out} =\begin{bmatrix}1 \\0 \\0 \\0 \\0\end{bmatrix}
$$

$$
f(text, pattern, out) = \text{return }text\text{ of 0s except with }out\text{ where }pattern\text{ in }text
$$

$$
f(\text{text}, \text{pattern}, \text{out}) = 
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

<p align="center"><i>Showing how the tokenization layer with a single neuron works on an input text</i></p>

$$
f_1(t, p_1, o_1) = 
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}, f_2(t, p_2, o_2) = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}, f_3(t, p_3, o_3) = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
, f_4(t, p_4, o_4) = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}, f_5(t, p_5, o_5) = 
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

$$
\text{layer output} = f_1(t, p_1, o_1) + f_2(t, p_2, o_2) + f_3(t, p_3, o_3) + f_4(t, p_4, o_4) + f_5(t, p_5, o_5) = 
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

<p align="center"><i>Showing how the multiple neurons in a tokenization layer come together to make the layer output</i></p>



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
