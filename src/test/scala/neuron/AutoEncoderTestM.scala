package neuron
import breeze.stats.distributions._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import neuron.math._
import neuron.autoencoder._

@RunWith(classOf[JUnitRunner])
class AutoEncoderTestM extends FunSuite with Optimizable {
	test("Test auto-encoder using NeuronMatrix") {
	  val inputDimension = 20
	  val hiddenDimension = 10
	  nn = new SimpleAutoEncoder(0.1, 0.1)(inputDimension,hiddenDimension)().create()
	  //nn = new SparseLinearAE(0.0,1.0,1.0)(inputDimension,hiddenDimension)().create()
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
