package tutorials

//import breeze.plot._
import breeze.stats.distributions._
import neuralnetwork._
import breeze.linalg._

// create custom Image AutoEncoder from SparseSingleLayerAE
class ImageAutoEncoder (val rowsMultCols:Int, override val hiddenDimension: Int) 
	extends SparseAutoEncoder (3.0, .0001) (rowsMultCols, hiddenDimension)(){
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
    weightsVector := raw.asWeight(NN.hiddenDimension, NN.rowsMultCols).transpose.vec(false)
    
    import java.io._
    printToFile(new File(filename))(p =>    
    for (i<- 0 until NN.hiddenDimension) { // display by hidden nodes
      val imgNull = new Weight(0,0)
      val img = new NeuronVector(NN.rowsMultCols)//
      weightsVector(imgNull, img)
      //println(img.vec.data)
      //p.println((img.data/norm(img.data)).data.mkString("\t")) // Just print
      p.println(img.normalized)
    })
  }
}

object ImageAutoEncoderTest extends Optimizable {
	def main(args: Array[String]): Unit = {
	  //val hidden = 25
	  //xData = LoadData.rawImages64()
	  val hidden = 500
	  xData = LoadData.mnistTrain()
	  
	  val numOfPixels = xData(0).length
	  yData = xData
	  
	  nn = new ImageAutoEncoder(numOfPixels, hidden).create() // the same

	  
	  val w = getRandomWeightVector()
	  var time:Long = 0
	  
	  time = System.currentTimeMillis();
	  val (obj, w2) = train(w, 100, L2Distance, 100, "sgd")
	  val (obj2, w3) = train(w2, 100, L2Distance, 1000, "sgd")
	  println(System.currentTimeMillis() - time, obj2)
	  
	  nn.asInstanceOf[InstanceOfImageAutoEncoder]
	  	.displayHiddenNetwork("data/UFLDL/sparseae/results500.txt")

	}
}