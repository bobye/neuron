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
   // create topology of neural network
  val a = new MaxoutSingleLayerNN(10, 5)
  val a2= new SingleLayerNeuralNetwork(20)
  val b = new RegularizedLinearNN(10,10, 0.001)
 
  val c = (a ** b).create()
  val d = (b ++ c) ** a2
  val e = (d * d).create() :+ 3
	
  test("Test feed-forward multi-perceptron using Array[NeuronVector]") {
	// setup Optimizable members
    nn = e.create(); println(nn); // print structure 
	
    gradCheck(1000, 1E-3)

  }
  
  test("Test feed-forward multi-perceptron using NeuronMatrix") {
	// setup Optimizable members
    nn = e.create(); println(nn); // print structure
	
    gradCheckM(1000, 1E-6)   

  }

}
