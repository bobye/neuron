package neuron.examples


import neuron.core._
import neuron.math._

/**
 * @author bobye
 */
object MLP_MNIST extends Workspace with Optimizable {
    def main(args: Array[String]): Unit = {
      /**      
      // you may try fractal neural network instead :)
      import neuron.misc._
      val fnn = new FractalNeuralNetwork(5, 
      new LinearNeuralNetwork(10,10) ** new SingleLayerNeuralNetwork(10))
    
      nn = (fnn ** new RegularizedLinearNN(784, fnn.inputDimension, 1E-4)).create()
      */
      
      // set @MLP=784-200-10, @weight_decay=1E-4
      nn = (new RegularizedLinearNN(200, 10, 1E-4) **
            new SingleLayerNeuralNetwork(200) **
            new RegularizedLinearNN(784, 200, 1E-4)).create() // nn is declared in trait @Optimizable
            
      // load standard MNIST training data
      val (xData, yData) = LoadData.mnistDataM("std", "train")
      
      // generate random weight and initialize
      val theta0 = nn.getRandomWeights("get random weights").toWeightVector()
      nn.setWeights("set weight", theta0);
      
      // full-batch training (@maxIter=200, @distance=SoftMaxDistance)
      val (_, theta) = trainx(xData, yData, theta0, 200, SoftMaxDistance)
      
      // load standard MNIST testing data
      val (xDataTest, yDataTest) = LoadData.mnistDataM("std", "t10k")
      
      // estimate accuracy
      val accuracy = (yDataTest.argmaxCol() countEquals nn(xDataTest, null).argmaxCol()) / xDataTest.cols.toDouble
      println(accuracy)
    }
}

/* Accuracy: 0.9806 */