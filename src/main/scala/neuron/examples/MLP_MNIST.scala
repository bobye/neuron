package neuron.examples

import neuron.core._
import neuron.math._
import breeze.stats.distributions._

object MLP_MNIST extends Workspace with Optimizable{

  var xDataM : NeuronMatrix = null
  var yDataM : NeuronMatrix = null
  
  var xDataMList: List[NeuronMatrix] = null
  var yDataMList: List[NeuronMatrix] = null
  var listOfRanges: List[Range] = null
  
  
  private def HamiltonianDynamics(x0: NeuronMatrix, 
		  						  y0: NeuronMatrix,
		  						  deltaT: Double,
		  						  ticks: Int,
		  						  alpha: Double): NeuronMatrix = {
    val x = x0.copy();
    val sigma2 = alpha*alpha
    val z = new NeuronMatrix(x.rows, x.cols, new Gaussian(0, alpha))
    //val cost0 = (z.euclideanSqrNormCol :/= (2*sigma2))
    val mem = initMemory();
    val f: NeuronMatrix => NeuronMatrix = {xx =>      
      val (cost1, zz1) = SoftMaxDistance.applyWithGradV(nn(xx, mem), y0);
      val (cost2, zz2) = L2Distance.applyWithGradV(xx, x0)

      val grad1 = nn.backpropagate(zz1 MultElemTransWith cost2, mem) 
      val grad2 = (zz2 MultElemTransWith cost1)
      println(grad1.euclideanSqrNormCol)
      println(grad2.euclideanSqrNormCol)
      grad1 += grad2
    }
    
    // leapfrog iteration
    z -= f(x) * (deltaT/2)
    for (i <- 0 until ticks) {
      x += z * (deltaT / sigma2)
      z -= f(x) * deltaT
    }
    x += z * (deltaT / (2*sigma2))
    //val u = new NeuronVector(x0.cols, new Uniform(0,1))
    /*
    val costN1 = (z.euclideanSqrNormCol :/= (2*sigma2))
    val costN2 = SoftMaxDistance.applyV(nn(x, mem), y0) *= L2Distance.applyV(x, x0)
    println(cost0)
    println(costN1)
    println(costN2)
    println(ExpFunction(cost0 -= costN1 -= costN2))
    */
    x
  }

  private def pListOfRanges(size: Int, blockSize: Int) = {
    val numOfBlock: Int = (size - 1)/blockSize + 1
    (0 until (numOfBlock-1)).toList.map(i => 
      blockSize*i until blockSize*(i+1)) :+ (blockSize*(numOfBlock-1) until size)
  }
  
  var batchCount: Int = 0
  var epochCount: Int = 0
  def getObjAndGradStoM (w: WeightVector, dw: WeightVector,
		  				 distance:DistanceFunction = L2Distance, 
		  				 batchSize: Int = 0): (Double, NeuronVector) = {
    val tildex = HamiltonianDynamics(xDataMList(batchCount), yDataMList(batchCount), 0.02, 10, 0.3)
    
    if (batchCount == 0) {
    import neuron.misc.IO
    IO.writeImage("tmp/sample-epoch" + epochCount)(
        IO.tile_raster_images(xDataMList(batchCount) padRow tildex, 
                              (28, 28*2),
                              (scala.math.sqrt(tildex.cols).toInt, scala.math.sqrt(tildex.cols).toInt),
                              (2, 2), false))    
    }
    val (cost, grad) = getObjAndGradM(tildex, yDataMList(batchCount), w, dw, distance, batchSize)

    //val (cost, grad) = getObjAndGradM(xDataMList(batchCount), yDataMList(batchCount), w, dw, distance, batchSize)

    batchCount = (batchCount + 1) % listOfRanges.size;
    (cost, grad)
  }  
  
  import breeze.linalg._
  import breeze.optimize._
  def trainxopt(w: WeightVector, 
		  		maxIter: Int = 40, 
		  		distance: DistanceFunction = L2Distance,
		  		batchSize: Int = 0): (Double, WeightVector) = {
    val dw = new WeightVector(w.length) // pre-allocate memory
    
    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val (obj, grad) = getObjAndGradStoM(new WeightVector(x), dw, distance, batchSize)
	    (obj, grad.data )
	  }    
    }
    var wopt: WeightVector = w;  
    
    for (i<- 0 until maxIter) {   
     epochCount = i; println("Epoch "+ epochCount)      
     
     listOfRanges = scala.util.Random.shuffle(pListOfRanges(xDataM.cols, batchSize))   
     xDataMList = listOfRanges.map(xDataM.Cols(_))
     yDataMList = listOfRanges.map(yDataM.Cols(_))  
    
      val sgdm  = new SGDmTrain(0.9, 0.1, listOfRanges.size)
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
    
    val (_, theta) = trainxopt(theta0, 30, SoftMaxDistance, batchSize)
    println((yDataM.argmaxCol().data :== nn(xDataM, null).argmaxCol().data).activeSize / xDataM.cols.toDouble)
    xDataM = null
    yDataM = null
    xDataMList = null
    yDataMList = null
    theta
  }
  
  
  def main(args: Array[String]): Unit = {
    // Create NN structure
    val L1 = new SingleLayerNeuralNetwork(800) ** new LinearNeuralNetwork(784,800)
    val L2 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(800,10)
    nn = (L2 ** L1).create()
    
    process(25, args(0), args(1))
        
    val dataTest = LoadData.mnistDataM(args(2), args(3))
    val xDataTestM = dataTest._1
    val yDataTestM = dataTest._2
    
    println((yDataTestM.argmaxCol().data :== nn(xDataTestM, null).argmaxCol().data).activeSize / xDataTestM.cols.toDouble)
    
  }

}