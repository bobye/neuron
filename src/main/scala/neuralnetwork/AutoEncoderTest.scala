package neuralnetwork
import breeze.stats.distributions._

object AutoEncoderTest extends Optimizable with Workspace{
	def main(args: Array[String]): Unit = {
	  val inputDimension = 20
	  val hiddenDimension = 10
	  //nn = new SingleLayerAutoEncoder()(20,10).create()
	  nn = new SparseSingleLayerAE(1.0,1.0)(inputDimension,hiddenDimension).create()
	  val numOfSamples = 100
	  xData = new Array(numOfSamples);
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1))  
	  }
	  yData = xData
	  
	  initMemory()
	  val amplitude = scala.math.sqrt(6.0/(inputDimension + hiddenDimension + 1.0))
	  val w = getRandomWeightVector(amplitude)
	  
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