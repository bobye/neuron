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
	xData = Array(new NeuronVector(nn.inputDimension, new Uniform(-1,1))) // only one sample
	yData = Array(new NeuronVector(nn.outputDimension, new Uniform(-1,1)))

	

	initMemory()
    val w = getRandomWeightVector(new Uniform(-1,1))	
	
	var (obj, grad) = getObjAndGrad(w)
	println(obj)

  }

}