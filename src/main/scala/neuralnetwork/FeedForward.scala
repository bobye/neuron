package neuralnetwork
import breeze.stats.distributions._

// This is an example about creating a mixed type feedforward neural network
// and how to get objective and gradient in terms of internal weights
object FeedForward extends Optimizable with Workspace{

  def main(args: Array[String]): Unit = {
	
    // create topology of neural network
	val a = new SingleLayerNeuralNetwork(SigmoidFunction, 10)
	val a2= new SingleLayerNeuralNetwork(SigmoidFunction, 20)
	val b = new LinearNeuralNetwork(10,10)
 
	val c = (a TIMES b).create()
	val d = (b PLUS c) TIMES a2
	val e = (d PLUS d) 
	
	// setup Optimizable members
	nn = e.create(); println(nn);
	val numOfSamples = 2
	xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  yData(i) = new NeuronVector(nn.outputDimension, new Uniform(-1,1))
	}
	

	

	initMemory()
    val w = getRandomWeightVector(new Uniform(-1,1))	
	
    
    // compute objective and gradient
    var time = System.currentTimeMillis();
	var (obj, grad) = getObjAndGrad(w)
	println(System.currentTimeMillis() - time, grad.data)
	
	//gradient checking
	time = System.currentTimeMillis()
	var grad2 = grad.copy()
	for (i<- 0 until w.length) {
	  val epsilon = 0.00001
	  val w2 = w.copy
	  w2.data(i) = w.data(i) + epsilon
	  val cost1 = getObj(w2)
	  w2.data(i) = w.data(i) - epsilon
	  val cost2 = getObj(w2)
	  
	  grad2.data(i) = (cost1 - cost2) / (2*epsilon)
	}
	println(System.currentTimeMillis() - time, grad2.data)

  }

}