package neuron

import neuron.core._
import neuron.autoencoder._
import neuron.math._
import neuron.misc._
import breeze.stats.distributions._
import breeze.linalg._
import breeze.optimize._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RNNTest extends FunSuite with Optimizable with Workspace with EncoderWorkspace {
    
    def fullBinaryTree(depth:Int) : BTree = {
      assert (depth <= 15 && depth >= 1)
      if (depth == 1) 
        new BLeaf()
      else
        new BBranch(fullBinaryTree(depth - 1), fullBinaryTree(depth -1))
    }
	
	test("test RNN") {
	  val wordLength = 10
	  val tree = fullBinaryTree(5)
	  val enc  = (new RecursiveSimpleAE(0.001, 0.1)(wordLength)).create()
	  val input = (new IdentityAutoEncoder(wordLength)).create()
	  val output = (new SingleLayerNeuralNetwork(1) ** new LinearNeuralNetwork(wordLength,1)).create()
	  
	  
	  nn = (output ** new RecursiveNeuralNetwork(tree, enc.extract(), input)).create() 
	  //nn = (output TIMES new RecursiveAutoEncoder(tree, enc, input, 1.0).encoCreate()).create()
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
	  
	  
	  gradCheck(1E-6)
	  
	  
	  val time = System.currentTimeMillis();
	  val (obj3, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj3, w2.data)
	  
	}
}
