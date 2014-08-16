package neuron
import neuron.core._
import neuron.math._
import breeze.stats.distributions._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DistanceTest extends FunSuite with Optimizable with Workspace {
  test("test distances") {
    val a = new SingleLayerNeuralNetwork(10)
    val b = new RegularizedLinearNN(10,10, 0.001)
    nn = (b ** a ** b).create()
    
    val numOfSamples = 100
	xData = new Array(numOfSamples); yData = new Array(numOfSamples)
	for (i<- 0 until numOfSamples) {
	  xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  yData(i) = new NeuronVector(nn.outputDimension, new Uniform(0,1))
	  //yData(i) :/= yData(i).sum
	}
	
	val w = gradCheck(1E-3, HistogramIntersectionKernelDistance)
	
	// train
	/*
	val time = System.currentTimeMillis()
	val (obj3, w2) = train(w, 100, HistogramIntersectionKernelDistance)
	println(System.currentTimeMillis() - time, obj3)
	* 
	*/
  }
  
  test("test batch distance") {
    //val a = new SingleLayerNeuralNetwork(2)
    val b = new LinearNeuralNetwork(2,2)
    nn = b.create()
    
    val numOfSamples = 200
	xDataM = new NeuronMatrix(nn.inputDimension, numOfSamples, new Uniform(-1,1))
    yDataM = new NeuronMatrix(nn.outputDimension, numOfSamples, new Uniform(0,1))
	
	val w = gradCheckM(1E-3, HistogramIntersectionKernelDistance)
	 
  }

}
