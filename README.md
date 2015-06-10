
neuron [![Build Status](https://travis-ci.org/bobye/neuron.svg?branch=master)](https://travis-ci.org/bobye/neuron)
========

This project is a _work-in-progress_, which provides a flexible framework to experiment and run neural network of heterogeneous topologies. A preliminary release will be visible soon. neuron is written in Scala which adopts the so-called ["define-by-run" scheme](https://github.com/bobye/neuron/wiki/Basics#design-rationale). 

Creator: [Jianbo Ye](http://www.personal.psu.edu/jxy198)

### Features
 - [template vs. module](https://github.com/bobye/neuron/wiki/Basics#template-vs-module)
 - neural network [operators](https://github.com/bobye/neuron/wiki/Basics#operators)
 - [autoencoders](https://github.com/bobye/neuron/wiki/Auto-Encoder) (w or w/o tiled weight)
 - activation functions: logistic, tanh, ReLU, softplus
 - metrics (L1, L2, Mahalanobis, Softmax)
 - regularization: weight decay, activation sparsity, dropout, maxout
 - data parallel framework: atomic parameters + distributed states
 - optimization: LBFGS, SGD, SAGD, SGD with momentum
 - recursive neural network

### Documentation
The simplest example to train a regularized multi-layer perceptron 
to predict handwritten digits from MNIST dataset. It takes around ten minutes.
```scala
package neuron.examples

import neuron.core._
import neuron.math._

object MLP_MNIST extends Workspace with Optimizable {
    def main(args: Array[String]): Unit = {
      // set @MLP=784-200-10, @weight_decay=1E-4
      nn = (new RegularizedLinearNN(200, 10, 1E-4) **
            new SingleLayerNeuralNetwork(200) **
            new RegularizedLinearNN(784, 200, 1E-4)).create() // nn is declared in trait @Optimizable
            
      // load standard MNIST training data
      val (xData, yData) = LoadData.mnistDataM("std", "train")
      
      // generate random weight and initialize
      val theta0 = nn.getRandomWeights("get random weights").toWeightVector()
      nn.setWeights("set weight", theta0);
      
      // full-batch training (@maxIter=200, @distance=SoftMaxDistance)
      val (_, theta) = trainx(xData, yData, theta0, 200, SoftMaxDistance)
      
      // load standard MNIST testing data
      val (xDataTest, yDataTest) = LoadData.mnistDataM("std", "t10k")
      
      // estimate accuracy
      val accuracy = (yDataTest.argmaxCol().data :== nn(xDataTest, null).argmaxCol().data)
                        .activeSize / xDataTest.cols.toDouble
      println(accuracy)
    }
}
/* Accuracy: 0.9806 */
```

Also have a look at
- [Basics](https://github.com/bobye/neuron/wiki/Basics): explains the most fundamental ideas for the use of neuron, and why they featured neuron as a good choice for prototyping neural networks that leverage flexibility, simplicity and efficiency. 
- [Auto-Encoder](https://github.com/bobye/neuron/wiki/Auto-Encoder): a special family of unsupervised neural network.
- [Examples](https://github.com/bobye/neuron/wiki/Examples): we have more examples under folder `src/main/scala/neuron/tutorials/`
- [Scaladoc](https://colourbrain.com/api/neuron): TBA

### FAQ

- _How is neuron different from other deep learning libraries (such as theano, torch7, etc), besides it is Scala based?_

  We argue that not only the number of parameters contributes to the representation ability of neural network, but also its infrastructure (network topology, train strategy, etc.) Neuron focuses on fast prototyping novel network architecture. Using Scala, we attempt to make the implementation of neural network in a mixed functional and imperative way ... though neuron is not at the mature shape to be industrial proven.

- _How is the speed of neruon?_

  Neuron is currently backed by [breeze](https://github.com/dlwh/breeze/) for numerical computation, which should be fast. And the extra cost for data control flow is minimized. Neuron provides convenient [data parallelization](https://github.com/bobye/neuron/wiki/Parallelization).

### Reference
* [Breeze](https://github.com/scalanlp/breeze/) and [Nak](https://github.com/scalanlp/nak): a set of libraries for machine learning and numerical computing
* [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial): a Stanford course, find solutions at [Github](https://github.com/search?q=UFLDL+Tutorial)
* [ScalaSTM](http://nbronson.github.io/scala-stm/):  a lightweight software transactional memory for Scala 

----
The MIT License (MIT)

Copyright (c) 2014 Jianbo Ye

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
