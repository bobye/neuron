package tutorials

import neuralnetwork._
import breeze.stats.distributions._
import breeze.linalg._
import breeze.optimize._



object RNNTest extends Optimizable with Workspace {
    
    def fullBinaryTree(depth:Int) : Tree = {
      assert (depth <= 15 && depth >= 1)
      if (depth == 1) 
        new Leaf()
      else
        new Branch(fullBinaryTree(depth - 1), fullBinaryTree(depth -1))
    }
	
	def main(args: Array[String]): Unit = {
	  val wordLength = 3
	  val tree = fullBinaryTree(10)
	  val enc  = (new RecursiveSimpleAE()(wordLength, 0.0, 0.1)).create()
	  val input = (new IdentityTransform(wordLength)).create()
	  val output = new SingleLayerNeuralNetwork(1) TIMES new LinearNeuralNetwork(wordLength,1)
	  
	  nn = (output TIMES new RecursiveNeuralNetwork(tree, enc, input)).create()
	  
	  val numOfSamples = 100
	  xData = new Array(numOfSamples)
	  yData = new Array(numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(nn.inputDimension, new Uniform(-1,1))  
	    yData(i) = new NeuronVector(1, new Uniform(-1,1))
	  }
	  
	  val w = getRandomWeightVector()
	  var time: Long = 0
	  
	  //val obj = getObj(w); println(obj)
	  
	  time = System.currentTimeMillis();
	  val (obj, grad) = getObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj, grad.data)
	  
	  // gradient checking
	  time = System.currentTimeMillis();
	  val (obj2, grad2) = getApproximateObjAndGrad(w)
	  println(System.currentTimeMillis() - time, obj2, grad2.data)
	  
	  
	  time = System.currentTimeMillis();
	  val (obj3, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj3)
	  
	}
}