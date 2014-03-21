package tutorials
import neuralnetwork._
import breeze.stats.distributions._

object SoftMaxUnitTest extends Optimizable with Workspace {
  def main(args: Array[String]): Unit = {
    val a = new SingleLayerNeuralNetwork(10)
    val b = new RegularizedLinearNN(10,10, 0.001)
    nn = (b TIMES a TIMES b).create()
    
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