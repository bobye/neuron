package tutorials

import neuralnetwork._
import breeze.stats.distributions._
import breeze.linalg._
import breeze.optimize._



object RNNTest extends Optimizable with Workspace with EncoderWorkspace {
    
    def fullBinaryTree(depth:Int) : Tree = {
      assert (depth <= 15 && depth >= 1)
      if (depth == 1) 
        new Leaf()
      else
        new Branch(fullBinaryTree(depth - 1), fullBinaryTree(depth -1))
    }
	
	def main(args: Array[String]): Unit = {
	  val wordLength = 10
	  val tree = fullBinaryTree(5)
	  val enc  = (new RecursiveSimpleAE()(wordLength, 0.001, 0.0)).create()
	  val input = (new IdentityAutoEncoder(wordLength)).create()
	  val output = (new SingleLayerNeuralNetwork(1) TIMES new LinearNeuralNetwork(wordLength,1)).create()
	  
	  
	  //nn = (output TIMES new RecursiveNeuralNetwork(tree, enc, input)).create() 
	  nn = (output TIMES new RecursiveAutoEncoder(tree, enc, input, 1.0).encoCreate()).create()
	  //nn = (enc TIMES enc).create()
	  
	  val w = getRandomWeightVector()
	  
	  val numOfSamples = 100
	  nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
	  
	  xData = new Array(numOfSamples)
	  yData = new Array(numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(0,1))  
	    yData(i) = nn(xData(i), initMemory()) //new NeuronVector(1, new Uniform(-1,1))
	  }
	  
	  
	  var time: Long = 0
	  
	  time = System.currentTimeMillis();
	  val (obj, grad) = getObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj, grad.data)
	  
	  // Gradient checking
	  time = System.currentTimeMillis();
	  val (obj2, grad2) = getApproximateObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj2, grad2.data)
	  
	  
	  time = System.currentTimeMillis();
	  val (obj3, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj3, w2.data)
	  
	}
}