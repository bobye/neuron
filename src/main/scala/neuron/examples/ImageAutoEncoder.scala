package neuron.tutorials

//import breeze.plot._
import breeze.stats.distributions._
import breeze.linalg._
import neuron.autoencoder._
import neuron.core._
import neuron.math._

// create custom Image AutoEncoder from SparseSingleLayerAE
class ImageAutoEncoder (val rowsMultCols:Int, override val hiddenDimension: Int, 
						val regularizedParam: Double, val sparsityParam: Double) 
	//extends SparseAutoEncoder (3.0, regularizedParam, 0.0, new KL_divergenceFunction(sparsityParam)) (rowsMultCols, hiddenDimension)()
	extends TiledWeightSparseAE (3.0, regularizedParam, 0.0, new KL_divergenceFunction(sparsityParam)) (rowsMultCols, hiddenDimension)()
{
  type Instance <: InstanceOfImageAutoEncoder
  override def create() = new InstanceOfImageAutoEncoder(this)
}
class InstanceOfImageAutoEncoder (override val NN: ImageAutoEncoder) 
	extends InstanceOfAutoEncoder(NN) //There is no InstanceOfSparseSingleLayerAE
{ 
  type Structure <: ImageAutoEncoder
  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    	val p = new java.io.PrintWriter(f)
    	try { op(p) } finally { p.close() }
  }
  
  def displayHiddenNetwork (filename: String) : Unit = { 
    val weightsVector = new WeightVector((NN.rowsMultCols)*NN.hiddenDimension)
    val raw = NN.inputLayer.W.vec() // getRandomWeights((System.currentTimeMillis()%100000).toString) // load in optimized weights
    weightsVector := raw.asNeuronMatrix(NN.hiddenDimension, NN.rowsMultCols).transpose.vec(false)
    
    import java.io._
    printToFile(new File(filename))(p =>    
    for (i<- 0 until NN.hiddenDimension) { // display by hidden nodes
      val img = new NeuronVector(NN.rowsMultCols)//
      weightsVector(null, img)
      p.println(img)
    })
  }
}

object ImageAutoEncoderTest extends Optimizable {
  
  override def getObjAndGradM (w: WeightVector, distance:DistanceFunction = L2Distance, batchSize: Int = 0): (Double, NeuronVector) = {
    val size = xDataM.cols
    assert(size >= 1 && (null == yDataM || size == yDataM.cols))
    val blockSize = 512
    val numOfBlock: Int = (size-1)/blockSize + 1
    val ranges = ((0 until (numOfBlock-1)).map(i => blockSize*i until blockSize*(i+1)) :+ (blockSize*(numOfBlock-1) until size)).par
    
    var totalCost:Double = 0.0
    
    val dw = new WeightVector(w.length)
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    ranges.map(r =>
    	nn(xDataM.Cols(r),initMemory())
    )
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    if (yDataM != null) {//supervised
      totalCost = ranges.map(r => {
        val mem = initMemory()
        val x = nn(xDataM.Cols(r), mem); val y = yDataM.Cols(r)
        val z = distance.grad(x, y)
        nn.backpropagate(z, mem) // update dw !
        distance(x,y)}).reduce(_+_)
    } else {//unsupervised
      ranges.map(r => {
        val mem = initMemory()
        val x = nn(xDataM.Cols(r), mem);
        nn.backpropagate(new NeuronMatrix(x.rows, x.cols), mem)
      })
      totalCost = 0.0
    }
    
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)
    //println(totalCost/size, regCost)
    (totalCost/size + regCost, dw/size)
  }
  
    object ioParam {
      
      val hidden = 25
	  xDataM = LoadData.rawImages64M()
      val hiddenUnitsFile = "data/UFLDL/sparseae/results25.txt"
      val regularizedParam = 0.0002
      val sparsityParam = 0.01  
      /*
	  val hidden = 200
	  xDataM = LoadData.mnistTrainM()
	  val hiddenUnitsFile = "data/UFLDL/sparseae/results500.txt"
      val regularizedParam = 0.005
      val sparsityParam = 0.1 
	  */  
	  val numOfPixels = xDataM.rows
	  yDataM = xDataM
    }
    
	def main(args: Array[String]): Unit = {
	  nn = new ImageAutoEncoder(ioParam.numOfPixels, ioParam.hidden, 
	      ioParam.regularizedParam, ioParam.sparsityParam).create() // the same

	  val w = getRandomWeightVector()
	  var time:Long = 0
	  
	  time = System.currentTimeMillis();
	  val (obj, w2) = trainx(w)
	  println(System.currentTimeMillis() - time, obj)
	  
	  nn.asInstanceOf[InstanceOfImageAutoEncoder]
	  	.displayHiddenNetwork(ioParam.hiddenUnitsFile)

	}
}
