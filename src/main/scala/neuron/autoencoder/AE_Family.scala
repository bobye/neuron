package neuron.autoencoder
import neuron.core._
import neuron.math._

class LinearAutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction,
    					 override val distance: DistanceFunction = L2Distance) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer), outputLayer, distance)	

class SimpleAutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))		

class Simple2AutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfLassoRegularizedLinearNN = 
	  new LassoRegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfLassoRegularizedLinearNN = 
	  new LassoRegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))	

class RobustAutoEncoder (lambda: Double = 0.0, regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRobustLinearNN = 
	  new RobustLinearNN(dimension, hiddenDimension, 0.1, lambda).create(),
	 val outputLayer: InstanceOfLinearNeuralNetwork = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))	
	  
class SparseLinearAE (val beta:Double = 0.0, // sparse penalty 
    					   lambda: Double = 0.0, // L2 regularization
    					   regCoeff: Double = 0.0, // autoencoding
    					   val penalty:NeuronFunction = new KL_divergenceFunction(0.01), // average activation
    					   val func: NeuronFunction = SigmoidFunction)
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern	
	extends AutoEncoder(regCoeff,
	    new ChainNeuralNetwork(new SparseSingleLayerNN(hiddenDimension, beta, func, penalty), inputLayer),
	    new RegularizedLinearNN(hiddenDimension, dimension, lambda))

class SparseAutoEncoder (val beta:Double = 0.0,
					  lambda:Double = 0.0,
					  regCoeff: Double = 0.0,
					  val penalty: NeuronFunction = new KL_divergenceFunction(0.01),
					  val func: NeuronFunction = SigmoidFunction)
	(dimension: Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern
	extends AutoEncoder(regCoeff, 
	    new ChainNeuralNetwork(new SparseSingleLayerNN(hiddenDimension, beta, func, penalty), inputLayer),
	    new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), 
	    					   new RegularizedLinearNN(hiddenDimension, dimension, lambda)))
	
class RecursiveLinearAE (func:NeuronFunction = SigmoidFunction) 
	(val wordLength: Int, lambda: Double = 0.0, regCoeff:Double = 0.0) 
	extends LinearAutoEncoder(lambda, regCoeff, func)(wordLength*2, wordLength)() 

class RecursiveSimpleAE (lambda: Double = 0.0, regCoeff:Double = 0.0, func:NeuronFunction =SigmoidFunction) 
	(val wordLength: Int)
	extends SimpleAutoEncoder(lambda, regCoeff, func)(wordLength*2, wordLength)()