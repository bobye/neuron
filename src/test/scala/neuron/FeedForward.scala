package neuron
import breeze.stats.distributions._
import neuron.core._
import neuron.math._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

// This is an example about creating a mixed type feedforward neural network
// and how to get objective and gradient in terms of internal weights
@RunWith(classOf[JUnitRunner])
class FeedForward extends FunSuite with Optimizable with Workspace{

  test("Test feed-forward multi-perceptron using Array[NeuronVector]") {
	
    // create topology of neural network
	val a = new MaxoutSingleLayerNN(10, 5)
	val a2= new SingleLayerNeuralNetwork(20)
	val b = new RegularizedLinearNN(10,10, 0.001)
 
	val c = (a ** b).create()
	val d = (b ++ c) ** a2
	val e = (d * d).create() :+ 3
	
	// setup Optimizable members
	//nn = (new RegularizedLinearNN(65,10, 0.001) TIMES (c TENSOR a3)).create() 
    nn = e.create(); println(nn); // print structure
	
	val numOfSamples = 1000
	xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  yData(i) = new NeuronVector(nn.outputDimension, new Uniform(-1,1))
	}
	
    val w = gradCheck(1E-6)
	
	// train
	val time = System.currentTimeMillis()
	val (obj3, w2) = train(w)
	println(System.currentTimeMillis() - time, obj3)
	//println(w.data)
	//println(w2.data)


  }

}
