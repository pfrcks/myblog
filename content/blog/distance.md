---
title: "Distance Metrics and Beyond"
date: 2018-11-28T13:52:38-05:00
draft: false
---

## Getting Started

A distance function(also referred as a metric) in mathematics refers to the function that defines a distance between any two points of a set. The most prevalent example is the Euclidean distance. But let's first define what makes a function a distance function.

For a function to be a distance function, it has to satisfy the following properties:

- Non-negativity : `$d(x, y) \geq 0$`
- Identity       : `$d(x, y) = 0 \iff x=y$`
- Symmetry       : `$d(x, y) = d(y, x)$`
- Triangle Inequality : `$d(x, z) \leq d(x, y) + d(y, z)$`

Some of the distance functions which we went over in class were `$L_p$` distances, Jaccard Distance, Cosine Distance and Edit distance. {TODO: refer class notes}. Now let's take a look at another kind of _distance_

## Kullback–Leibler divergence

To call KL divergence a distance is like calling tomato a vegetable. KL divergence is used to measure the difference between two probability distributions over the same random variable x.

`$$D_{KL}(p(x)||q(x)) = \sum_{x \in X} p(x) \ln \frac{p(x)}{q(x)}$$`

In terms of information theory, KL divergence measures the expected number of extra bits required to code samples from distribution `$p(x)` when using code based on `$q(x)$`. Another way to think about this is that `$p(x)` is the **true** distribution of data and `$q(x)$` is our **approximation** of it.

Even though it looks like KL divergence measures **distance** between two probability distributions, it is not a distance measure. KL divergence is neither symmetric not does it satisfy the triangle inequality. It is however non-negative.

If it is not a distance metric, then how is it used? In the formula presented above, the true distribution is often intractable. Therefore by minimizing the KL divergence by using tractable approximated distributions we can get an approximate distribution from which samples can be drawn.

## Distance metrics in the wild

As expected Euclidean distance is perhaps the most used distance metric. We commonly utilize it for ML algorithms like KNN, K-Means, etc. Cosine distance is another heavily utilized metric used prominently in calculating similarity in NLP based tasks. But how does the choice of distance metric effect the model?

## Curse of Dimensionality

A typical supervised machine learning problem involves examples which are essentially vectors in a high dimensional feature space. A feature vector `$x=<x_1, x_2, x_3, ..., x_d>$`, can be said to be belonging to the space `$\mathbb{R}^D$`

Recall the concepts underlying K-Nearest Neighbors. We utilize the labels of the K neares points to make prediction about the label of the point under consideration. This is done because of our inductive bias that if a point lies in a space, then it is similar to other points which are in vicinity to it. Each dimension of the feature vector is taken into consdiration when the concept of **nearest** is being applied. However this introduces a couple of problems. Since KNN is giving importance to all features equally, in a dataset containing only a select relevant features KNN can perform poorly.

Another problem which arises is of feature scaling. Let's assume that the distance metric for KNN is Euclidean distance. In such a case different scales amongst the dimensions of the feature vector can result in some dimensions being completely ignored. However, this has an easy fix. We can scale the dimensions to zero mean and unit variance and get rid of this issue.

However, the problem becomes much more severe when the number of dimensions start increasing. 