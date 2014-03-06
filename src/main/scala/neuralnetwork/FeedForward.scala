package neuralnetwork

object FeedForward extends Workspace{

  def main(args: Array[String]): Unit = {
	def inputDimension = 10
	def outputDimension = 10
	
	val a = new SingleLayerNeuralNetwork(IndentityFunction, 10)
	val b = new LinearNeuralNetwork(10,10)
 
	val c = (a TIMES b).create()
	val d = (b PLUS c) TIMES a
	val e = (d PLUS d)
	val f = e.create()
	
    println(f.getWeights("1").data.length) 
    
    
    val s = new RecursiveSingleLayerCAE(SigmoidFunction)(10, 7).create()
    println(s.getWeights("2").data.length)
    
    val r = new InstanceOfEncoderNeuralNetwork (s)
	println(r.getWeights("3").data.length)

  }

}