package neuron.examples

import neuron.core._
import neuron.math._

object MLP_MNIST extends Workspace with Optimizable{

  var xDataM : NeuronMatrix = null
  var yDataM : NeuronMatrix = null
  
  var xDataMList: List[NeuronMatrix] = null
  var yDataMList: List[NeuronMatrix] = null
  var listOfRanges: List[Range] = null

  private def pListOfRanges(size: Int, blockSize: Int) = {
    val numOfBlock: Int = (size - 1)/blockSize + 1
    (0 until (numOfBlock-1)).toList.map(i => 
      blockSize*i until blockSize*(i+1)) :+ (blockSize*(numOfBlock-1) until size)
  }
  
  var batchCount: Int = 0
  def getObjAndGradStoM (w: WeightVector, 
		  				 distance:DistanceFunction = L2Distance, 
		  				 batchSize: Int = 0): (Double, NeuronVector) = {
    val (cost, grad) = getObjAndGradM(xDataMList(batchCount), yDataMList(batchCount), w, distance, batchSize)
    batchCount = (batchCount + 1) % listOfRanges.size;
    (cost, grad)
  }  
  
  import breeze.linalg._
  import breeze.optimize._
  def trainxopt(w: WeightVector, 
		  		maxIter: Int = 40, 
		  		distance: DistanceFunction = L2Distance,
		  		batchSize: Int = 0): (Double, WeightVector) = {
    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val (obj, grad) = getObjAndGradStoM(new WeightVector(x), distance, batchSize)
	    (obj, grad.data )
	  }    
    }
    var wopt: WeightVector = w;  
    
    for (i<- 0 until maxIter) {    
      println("Epoch "+ i)      
     listOfRanges = scala.util.Random.shuffle(pListOfRanges(xDataM.cols, batchSize))   
     xDataMList = listOfRanges.map(xDataM.Cols(_))
     yDataMList = listOfRanges.map(yDataM.Cols(_))  
    
      val sgdm  = new SGDmTrain(0.0, 0.1, listOfRanges.size)
      wopt = new WeightVector(sgdm.minimize(f, wopt.data))
    }
    
    (f(wopt.data), wopt)
  }
  def process(batchSize: Int = 0, setType: String, setMeta: String): WeightVector = {    
    val theta0 = nn.getRandomWeights("get random weights").toWeightVector()
    // Load Dataset
    val data = LoadData.mnistDataM(setType, setMeta)
    
    xDataM = data._1
    yDataM = data._2
    
    val (_, theta) = trainxopt(theta0, 40, SoftMaxDistance, batchSize) 
    xDataM = null
    yDataM = null
    xDataMList = null
    yDataMList = null
    theta
  }
  
  
  def main(args: Array[String]): Unit = {
    // Create NN structure
    val L1 = new SingleLayerNeuralNetwork(200) ** new LinearNeuralNetwork(784,1000)
    val L2 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(200,10)
    nn = (L2 ** L1).create()
    
    process(10, args(0), args(1))
        
    val dataTest = LoadData.mnistDataM(args(2), args(3))
    val xDataTestM = dataTest._1
    val yDataTestM = dataTest._2
    
    println((yDataTestM.argmaxCol().data :== nn(xDataTestM, null).argmaxCol().data).activeSize / xDataTestM.cols.toDouble)
    
  }

}