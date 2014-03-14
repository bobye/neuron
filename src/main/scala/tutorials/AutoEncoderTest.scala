package tutorials
import breeze.stats.distributions._
import neuralnetwork._

object AutoEncoderTest extends Optimizable {
	def main(args: Array[String]): Unit = {
	  val inputDimension = 20
	  val hiddenDimension = 10
	  nn = new LinearAutoEncoder()(inputDimension,hiddenDimension).create()
	  //nn = new SparseLinearAE(1.0,1.0)(inputDimension,hiddenDimension).create()
	  val numOfSamples = 100
	  xData = new Array(numOfSamples);
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1))  
	  }
	  yData = xData
	  
	  val w = getRandomWeightVector()
	  
	  var time = System.currentTimeMillis();
	  val (obj, grad) = getObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj, grad.data)
	  
	  // gradient checking
	  time = System.currentTimeMillis();
	  val (obj2, grad2) = getApproximateObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj2, grad2.data)
	  
	  
	  time = System.currentTimeMillis();
	  val (obj3, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj3)
	  println(w.data)
	  println(w2.data)
	  

	}
}