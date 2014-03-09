package neuralnetwork
import breeze.stats.distributions._

object AutoEncoderTest extends Optimizable with Workspace{
	def main(args: Array[String]): Unit = {
	  //nn = new SingleLayerAutoEncoder()(20,10).create()
	  nn = new SparseSingleLayerAE(0.)(20,10).create()
	  val numOfSamples = 100
	  xData = new Array(numOfSamples);
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1)) 
	    /* 
	     * It is interesting to see the difference between the above line
	     * If xData are sampled from the same random generator, it is essentially not random
	     * a Neural Network can capture the patterns very well
	     */
	    // xData(i) = new NeuronVector(nn.inputDimension) 
	  }
	  yData = xData
	  
	  initMemory()
	  val w = getRandomWeightVector()
	  
	  var time = System.currentTimeMillis();
	  val (obj, grad) = getObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj, grad.data)
	  
	  time = System.currentTimeMillis();
	  val (obj2, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj2)
	  
	  //println(w2.data)
	  
	}
}