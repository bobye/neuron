package neuron
import neuron.core._
import neuron.math._
import breeze.stats.distributions._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class SoftMaxUnitTest extends FunSuite with Optimizable with Workspace {
  test("test softmax") {
    val a = new SingleLayerNeuralNetwork(10)
    val b = new RegularizedLinearNN(10,10, 0.001)
    nn = (b ** a ** b).create()
    
    val numOfSamples = 100
	xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  yData(i) = new NeuronVector(nn.outputDimension, new Uniform(0,1))
	  yData(i) :/= yData(i).sum
	}
	
	val w = getRandomWeightVector()	
	// compute objective and gradient
    var time = System.currentTimeMillis();
	val (obj, grad) = getObjAndGrad(w, SoftMaxDistance)
	println(System.currentTimeMillis() - time, obj, grad.data)
	

	// gradient checking
	time = System.currentTimeMillis()
    val (obj2, grad2) = getApproximateObjAndGrad(w, SoftMaxDistance)
	println(System.currentTimeMillis() - time, obj2, grad2.data)
	
	// train
	time = System.currentTimeMillis()
	val (obj3, w2) = train(w)
	println(System.currentTimeMillis() - time, obj3)
	println(w.data)
	println(w2.data)
  }

}
