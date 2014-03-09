package neuralnetwork
import breeze.stats.distributions._

object AutoEncoderTest extends Optimizable with Workspace{
	def main(args: Array[String]): Unit = {
	  nn = new SingleLayerAutoEncoder(SigmoidFunction)(20,10).create()
	  val numOfSamples = 100
	  xData = new Array(numOfSamples);
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	  }
	  yData = xData
	  
	  initMemory()
	  val w = getRandomWeightVector(new Uniform(-1,1))
	  
	  var time = System.currentTimeMillis();
	  val (obj, _) = getObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj)
	
	  time = System.currentTimeMillis();
	  val (obj2, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj2)
	  //println(w2.data)
	  
	}
}