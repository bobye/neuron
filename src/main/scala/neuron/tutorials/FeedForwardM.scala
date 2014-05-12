package neuron.tutorials
import breeze.stats.distributions._
import neuron.core._
import neuron.math._

// This is an example about creating a mixed type feedforward neural network
// and how to get objective and gradient in terms of internal weights
object FeedForwardM extends Optimizable with Workspace{

  def main(args: Array[String]): Unit = {
	
    // create topology of neural network
	val a = new SingleLayerNeuralNetwork(10)
	val a2= new SingleLayerNeuralNetwork(20)
	val a3 = new SingleLayerNeuralNetwork(5)
	val b = new RegularizedLinearNN(10,10, 0.001)
 
	val c = (a TIMES b).create()
	val d = (b PLUS c) TIMES a2
	val e = (d PLUS d) 
	
	// setup Optimizable members
    nn = e.create(); println(nn); // print structure
	
	val numOfSamples = 1000
	xDataM = new NeuronMatrix(nn.inputDimension, numOfSamples, new Uniform(-1,1))
	yDataM = new NeuronMatrix(nn.outputDimension, numOfSamples, new Uniform(-1,1))
	
    val w = getRandomWeightVector()		
    
    // compute objective and gradient
    var time = System.currentTimeMillis();
	val (obj, grad) = getObjAndGradM(w)
	println(System.currentTimeMillis() - time, obj, grad.data)
	

	// gradient checking
	time = System.currentTimeMillis()
    val (obj2, grad2) = getApproximateObjAndGradM(w)
	println(System.currentTimeMillis() - time, obj2, grad2.data)
	
	
	// train
	time = System.currentTimeMillis()
	val (obj3, w2) = trainx(w)
	println(System.currentTimeMillis() - time, obj3)
	println(w.data)
	println(w2.data)

  }

}
