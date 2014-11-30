package neuron.examples

import neuron.core._
import neuron.math._
import breeze.stats.distributions._
import breeze.linalg.reshape

object MLP_MNIST extends Workspace with Optimizable{

  var xDataM : NeuronMatrix = null
  var yDataM : NeuronMatrix = null
  var xDataMValid: NeuronMatrix = null
  var yDataMValid: NeuronMatrix = null
  
  var xDataMList: List[NeuronMatrix] = null
  var yDataMList: List[NeuronMatrix] = null
  var listOfRanges: List[Range] = null
  
  private def RandomWalkDynamics(x0: NeuronMatrix, y0: NeuronMatrix, alpha: Double, copies: Int): 
	  (NeuronMatrix, NeuronMatrix) = {
    val z = new NeuronMatrix(x0.rows * x0.cols, copies, new Gaussian(0, alpha))
    val w = new NeuronMatrix(y0.rows * y0.cols, copies)
    ((z AddWith x0.vec(false)).reshape(x0.rows, x0.cols * copies),
     (w AddWith y0.vec(false)).reshape(y0.rows, y0.cols * copies))
  }	
  private def HamiltonianDynamics(x0: NeuronMatrix, 
		  						  y0: NeuronMatrix,
		  						  deltaT: Double,
		  						  ticks: Int,
		  						  alpha: Double,
		  						  copies: Int): NeuronMatrix = {
    val x = x0.copy();
    val sigma2 = alpha*alpha
    val z = new NeuronMatrix(x.rows, x.cols, new Gaussian(0, alpha))
    //val cost0 = (z.euclideanSqrNormCol :/= (2*sigma2))
    val mem = initMemory();
    val f: NeuronMatrix => NeuronMatrix = {xx =>      
      val (cost1, zz1) = SoftMaxDistance.applyWithGradV(nn(xx, mem), y0);
      val (cost2, zz2) = L2Distance.applyWithGradV(xx, x0)
      //val costex = ExpFunction(cost1)

      val grad1 = nn.backpropagate(zz1 MultElemTransWith cost2, mem) :*= 100
      val grad2 = (zz2 MultElemTransWith cost1)
      //println(grad1.euclideanSqrNormCol)
      //println(grad2.euclideanSqrNormCol)
      grad1 += grad2
    }
    
    // leapfrog iteration
    val beta: Double  = 10.0
    z -= f(x) * (deltaT * beta/2)
    for (i <- 0 until ticks) {      
      x += z * (deltaT / sigma2)
      z -= f(x) * (deltaT * beta)     
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

  /*
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
    /*
    val tildex = HamiltonianDynamics(xDataMList(batchCount), yDataMList(batchCount), 0.02, 10, 3.0)
    //val tildex = RandomWalkDynamics(xDataMList(batchCount), 0.1)
    val (cost, grad) = getObjAndGradM(tildex, yDataMList(batchCount), w, dw, distance, batchSize)
    if (batchCount == 0) {
    import neuron.misc.IO
    IO.writeImage("tmp/sample-epoch" + epochCount)(
        IO.tile_raster_images(tildex, 
                              (28, 28),
                              (scala.math.sqrt(tildex.cols).toInt, scala.math.sqrt(tildex.cols).toInt),
                              (2, 2), true))
    IO.writeImage("tmp/filter-epoch" + epochCount)(
        IO.tile_raster_images(L1.second.W.transpose, 
                              (28, 28),
                              (scala.math.sqrt(L1.second.W.rows).toInt, scala.math.sqrt(L1.second.W.rows).toInt),
                              (2, 2), true))                              
    }
    */

    val (cost, grad) = getObjAndGradM(xDataMList(batchCount), yDataMList(batchCount), w, dw, distance, batchSize)

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
    var acc = 0.0;
    var bestacc = 0.0
    var count = 0;
    
    import scala.util.control.Breaks._
    breakable { 
    for (i<- 0 until maxIter) {   
     epochCount = i; println("Epoch "+ epochCount)      
     
     listOfRanges = scala.util.Random.shuffle(pListOfRanges(xDataM.cols, batchSize))   
     xDataMList = listOfRanges.map(xDataM.Cols(_))
     yDataMList = listOfRanges.map(yDataM.Cols(_))  
    
      val sgdm  = new SGDmTrain(0.9, 0.1, listOfRanges.size)
      wopt = new WeightVector(sgdm.minimize(f, wopt.data))
      acc = (yDataMValid.argmaxCol().data :== nn(xDataMValid, null).argmaxCol().data).activeSize / xDataMValid.cols.toDouble
      if (acc > bestacc || i<10) {
        bestacc = acc;
        count = 0;
      } else {
        count = count + 1;
      }
      println(acc, count)
      if (count > 20) break;
    }}
    
    (f(wopt.data), wopt)
  }
  */
  def process(batchSize: Int = 0, setType: String, setMeta: String): WeightVector = {    
    val theta0 = nn.getRandomWeights("get random weights").toWeightVector()
    //theta0.importFromFile("weights-pre.txt")
    // Load Dataset
    val data = LoadData.mnistDataM(setType, setMeta)
    
    xDataM = data._1.Cols(0 until 1000)
    yDataM = data._2.Cols(0 until 1000)
    
    import neuron.misc.IO
    IO.writeImage("tmp/samples")(
        IO.tile_raster_images(xDataM, 
                              (28, 28),
                              (20, 20),
                              (2, 2), true))     
    val co = RandomWalkDynamics(xDataM, yDataM, 0.5, 64) 
    //val co = HamiltonianDynamics(xDataM, yDataM, 0.02, 100, 3.0)
    xDataM = co._1
    yDataM = co._2
    import neuron.misc.IO
    IO.writeImage("tmp/corrupted_samples")(
        IO.tile_raster_images(xDataM, 
                              (28, 28),
                              (20, 20),
                              (2, 2), true))    
    
    //xDataMValid = data._1.Cols(10000 until 12000)
    //yDataMValid = data._2.Cols(10000 until 12000)
     
    
    val (_, theta) = trainx(xDataM, yDataM, theta0, 500, SoftMaxDistance)
    println((yDataM.argmaxCol().data :== nn(xDataM, null).argmaxCol().data).activeSize / xDataM.cols.toDouble)
    xDataM = null
    yDataM = null
    xDataMValid = null
    yDataMValid = null
    xDataMList = null
    yDataMList = null
    theta
  }
  
  val L1 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(784,10).create()
  //val L2 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(100,10).create()  
  def main(args: Array[String]): Unit = {
    // Create NN structure

    nn = L1.create() // (L2 ** L1).create()
    
    process(500, args(0), args(1))
        
    val dataTest = LoadData.mnistDataM(args(2), args(3))
    val xDataTestM = dataTest._1
    val yDataTestM = dataTest._2
    
    println((yDataTestM.argmaxCol().data :== nn(xDataTestM, null).argmaxCol().data).activeSize / xDataTestM.cols.toDouble)
    import java.io._
    import neuron.misc.IO
    IO.printToFile(new File("weights.txt"))(p => 
          p.println(nn.getWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString))) 
  }

}