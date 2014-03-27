package tutorials
import neuralnetwork._
import breeze.stats.distributions._

// This code is simple test unit for converting a chain to binary tree in a greedy way
object GreedyMergeGraphTest extends Optimizable with Workspace {
  def f (x: NeuronVector, y: NeuronVector) : (Double, NeuronVector) = {
    (x.sum + y.sum, x+y)
  }
  
  object tgmc extends GreedyMergeChain(f)
  
  def main(args: Array[String]): Unit = {
    val numOfSamples = 100
	  xData = new Array(numOfSamples)
    for (i<- 0 until numOfSamples) {
      xData(i) = new NeuronVector(10, new Uniform(-1,1)) 
    }
    
    tgmc.loadChain(xData)
    tgmc.greedyMerge()// tgmc.nodes is set of trees
  }

}