package neuron.tutorials
import breeze.stats.distributions._
import neuron.math._
import neuron.autoencoder._

object AutoEncoderTestM extends Optimizable {
	def main(args: Array[String]): Unit = {
	  val inputDimension = 20
	  val hiddenDimension = 10
	  //nn = new SimpleAutoEncoder()(inputDimension,hiddenDimension,0.1, 0.1).create()
	  nn = new SparseLinearAE(0.0,1.0,1.0)(inputDimension,hiddenDimension)().create()
	  //nn = new SparseLinearAE(1.0,1.0,1.0)(inputDimension,hiddenDimension)().create() // Gradient check succeed
	  
	  val numOfSamples = 100
	  xDataM = new NeuronMatrix(nn.inputDimension, numOfSamples, new Uniform(-1,1))
	  yDataM = xDataM
	  
	  val w = getRandomWeightVector()
	  
	  var time = System.currentTimeMillis();
	  val (obj, grad) = getObjAndGradM(w)
	  println(System.currentTimeMillis() - time, obj, grad.data)
	  
	  // gradient checking
	  time = System.currentTimeMillis();
	  val (obj2, grad2) = getApproximateObjAndGradM(w)
	  println(System.currentTimeMillis() - time, obj2, grad2.data)
	  
	  
	  time = System.currentTimeMillis();
	  val (obj3, w2) = trainx(w)
	  println(System.currentTimeMillis() - time, obj3)
	  println(w.data)
	  println(w2.data)
	  

	}
}