# Basics about neuron
In this page, we will explain the basic concepts of neuron and how neuron works.

### Why neuron?
Neural Network (NN) has received renewed attention in recent years in the area of machine learning, computer vision, and signal processing. The revival of Neural Network attributes to its flexibilities in designing heterogeneous architectures, customizing objective, and choosing training tricks. There already exist many libraries that partially support deep learning, for example, ...

Neuron, as yet another alternative for deep learning, was created for my personal research, but also aimed at more general deep learning practitioners. It does not focus on one or two types of neural network structures, but provides mechanism for designing novel structures. This means, it may not be computational optimized for certain tasks, such as convolutional neural network or so. 

### Numerical Facilities
Neuron currently uses [breeze](https://github.com/scalanlp/breeze) for numerical operations of vectors and matrices. It has two classes, NeuronVector and NeuronMatrix, which interfaces to DenseVector[Double] and DenseMatrix[Double] of breeze. For more information, please checkout [LinearAlgebra.scala](../src/main/scala/neuron/math/LinearAlgebra.scala)

### Composition of Neural Networks
A user of neuron will be able to compose neural networks on his/her own. The fundamental concept is the distinction between template and module, which are `NeuralNetwork` class and `InstanceOfNeuralNetwork` class respectively. They both are inherited from `Operational` class. 

`NeuralNetwork` class is a template class that defines the type and hyper-parameters of a building block. For example, `SingleLayerNeuralNetwork` specifies the activation function (e.g. sigmoid, tanh, ...) for element-wise nonlinear transform. It has a member function `create()` which returns an instance of given template, as type of `InstanceOfNeuralNetwork`, that instantiates the internal parameters. For example, `LinearNeuralNetwork::create()` instantiates the internal weight and bias of a linear NN layer, which are data members of `InstanceOfLinearNeuralNetwork` class.

The distinction among template and module, and mixtures of them (as we will see), provides the possibilities for sharing and reusing parameters. Let's look at the following example:
```scala
// a is template, b, c are modules
val a = new LinearNeuralNetwork(10,5) // inputDimension: 10; outputDimension: 5
val b = new SingleLayerNeuralNetwork(5).create()
val c = new LinearNeuralNetwork(5,10).create() 

val d = (c TIMES b TIMES a) // type Operational
val e = (d PLUS d).create()
println(e) //print network structures by IDs of building blocks
```
Here `a` serves as a template building block in `d`, while `b` and `c` are instantiated modules in `d`. By duplicating `d` in building `e`, modules are copied by references and template are copied in twice that yields two distanctive modules of `a` by `(d PLUS d).create()`. Therefore, it is equivalent as follows
```
val e = ((c TIMES b TIMES a.create()) PLUS (c TIMES b TIMES a.create())).create()
``` 
where `a.create()` are called twice. 


