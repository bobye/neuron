package tutorials

//import breeze.plot._
import breeze.stats.distributions._
import neuralnetwork._
import breeze.linalg._

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
    object ioParam {
      
      val hidden = 25
	  xDataM = LoadData.rawImages64M()
      val hiddenUnitsFile = "data/UFLDL/sparseae/results25.txt"
      val regularizedParam = 0.0001
      val sparsityParam = 0.01  
      /*
	  val hidden = 200
	  xDataM = LoadData.mnistTrainM()
	  val hiddenUnitsFile = "data/UFLDL/sparseae/results500.txt"
      val regularizedParam = 0.003
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