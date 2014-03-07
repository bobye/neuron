package neuralnetwork
import breeze.stats.distributions._

object FeedForward extends Workspace{

  def main(args: Array[String]): Unit = {
	def inputDimension = 10
	def outputDimension = 10
	
	val a = new SingleLayerNeuralNetwork(SigmoidFunction, 10)
	val a2= new SingleLayerNeuralNetwork(SigmoidFunction, 20)
	val b = new LinearNeuralNetwork(10,10)
 
	val c = (a TIMES b).create()
	val d = (b PLUS c) TIMES a2
	val e = (d PLUS d)
	val f = e.create()
	//val f = a.create()
	
	
	//println(f)
	f.init("f").allocate("f")
	
    val wvlength = f.getWeights("1").length // get dimension of weights
    val wv  = new WeightVector(wvlength)
	wv.set(new NeuronVector(wvlength, new Uniform(-1,1))) // initialize randomly
	f.setWeights("1s", wv)
	val input = new NeuronVector(f.inputDimension, new Uniform(-1,1))
	val output = new NeuronVector(f.outputDimension, new Uniform(-1,1))
	println(input.length)
	println(f.backpropagate(f(input) - output))
    
    
    val s = new RecursiveSingleLayerCAE(SigmoidFunction)(10, 7).create()
    println(s.getWeights("2").length)
    
    val r = new InstanceOfEncoderNeuralNetwork (s)
	println(r.getWeights("3").length)


  }

}