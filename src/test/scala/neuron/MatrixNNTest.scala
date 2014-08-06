package neuron
import breeze.stats.distributions._
import neuron.core._
import neuron.math._
import neuron.unstable._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class MatrixNNTest extends FunSuite with Optimizable with Workspace {
  test("test matrix network") {
    val inputTensorDimension = 10
    val outputTensorDimension= 10
    //nn = new BiLinearSymmetricNN(inputTensorDimension,outputTensorDimension).create()
    val a = new RegularizedBiLinearSymNN(inputTensorDimension,outputTensorDimension, 0.1).create()
    nn = (a ** a).create()
    val numOfSamples = 1
    xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  val xIn = new NeuronVector(inputTensorDimension, new Uniform(-1,1))
	  val yOut= new NeuronVector(outputTensorDimension, new Uniform(-1,1))
	  xData(i) =  (xIn CROSS xIn).vec()
	  yData(i) =  (yOut CROSS yOut).vec()
	}
    gradCheck(1E-6)
  }
}
