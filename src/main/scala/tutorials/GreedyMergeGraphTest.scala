package tutorials
import neuralnetwork._
import breeze.stats.distributions._

// This code is simple test unit for converting a chain to binary tree in a greedy way
// Next step to implement Recursive Auto-Encoder Paper:
//     Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions
object GreedyMergeGraphTest extends Optimizable with Workspace {
  def f (x: NeuronVector, y: NeuronVector) : (Double, NeuronVector) = {
    (x.sum + y.sum, x+y)
  }
  
  object tgmc extends GreedyMergeChain(f)
  
  def main(args: Array[String]): Unit = {
    val wordLength = 10
    val chainLength= 10
    val numOfSamples = 100
	xData = new Array(numOfSamples)
    yData = new Array(numOfSamples)
    for (i<- 0 until numOfSamples) {
      xData(i) = new NeuronVector(wordLength*chainLength, new Uniform(-1,1)) 
      yData(i) = new NeuronVector(1, new Uniform(-1,1))
    }
    
    tgmc.loadChain(xData(0), wordLength)
    tgmc.greedyMerge()// tgmc.nodes is set of trees
    val node = tgmc.nodes.iterator.next
    
    val enc  = (new RecursiveSimpleAE()(wordLength, 0.0, 0.1)).create()
	val input = (new IdentityTransform(wordLength)).create()
	val output = new SingleLayerNeuralNetwork(1) TIMES new LinearNeuralNetwork(wordLength,1)
    
    nn = (output TIMES new RecursiveNeuralNetwork(node.t, enc, input)).create()
    
    val w = getRandomWeightVector()
    var time = System.currentTimeMillis();
	val (obj3, w2) = train(w)
	println(System.currentTimeMillis() - time, obj3)
	
  }

}