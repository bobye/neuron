package neuron.tutorials
import breeze.stats.distributions._
import neuron.core._
import neuron.math._

// This is an example about creating a mixed type feedforward neural network
// and how to get objective and gradient in terms of internal weights
object FeedForward extends Optimizable with Workspace{

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
	//nn = (new RegularizedLinearNN(65,10, 0.001) TIMES (c TENSOR a3)).create() 
    nn = e.create(); println(nn); // print structure
	
	val numOfSamples = 1000
	xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  yData(i) = new NeuronVector(nn.outputDimension, new Uniform(-1,1))
	}
	
    val w = getRandomWeightVector()		
    
    // compute objective and gradient
    var time = System.currentTimeMillis();
	val (obj, grad) = getObjAndGrad(w)
	println(System.currentTimeMillis() - time, obj, grad.data)
	

	// gradient checking
	time = System.currentTimeMillis()
    val (obj2, grad2) = getApproximateObjAndGrad(w)
	println(System.currentTimeMillis() - time, obj2, grad2.data)
	
	// train
	time = System.currentTimeMillis()
	val (obj3, w2) = train(w)
	println(System.currentTimeMillis() - time, obj3)
	println(w.data)
	println(w2.data)


  }

}
