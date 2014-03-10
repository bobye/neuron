package neuralnetwork

//import breeze.plot._
import breeze.linalg._
import breeze.stats.distributions._

class ImageVector (override val data: DenseVector[Double]) extends NeuronVector(data) {
  def this(rows: Int, cols: Int) = this(DenseVector.zeros[Double] (rows * cols))
  def this(rows: Int, cols: Int, rand: Rand[Double]) = this(DenseVector.rand(rows * cols, rand))
  def saveas(str: String): Unit = {// write image
    //val f = Figure()
    println(data)
  }
}

class ImageAutoEncoder (val rows:Int, val cols:Int, override val hiddenDimension: Int) 
	extends SparseSingleLayerAE (1.0) (rows*cols, hiddenDimension){
  type Instance <: InstanceOfImageAutoEncoder
  override def create() = new InstanceOfImageAutoEncoder(this)
}
class InstanceOfImageAutoEncoder (override val NN: ImageAutoEncoder) 
	extends InstanceOfAutoEncoder(NN) {
  type Structure <: ImageAutoEncoder
  def displayHiddenNetwork (str:String) : Null = { 
    val weightsVector = new WeightVector((NN.rows*NN.cols+1)*NN.hiddenDimension)
    weightsVector := inputLayer.getWeights((System.currentTimeMillis()%100000).toString) // load in optimized weights
    for (i<- 0 until NN.hiddenDimension) { // display by hidden nodes
      val img = new ImageVector(NN.rows, NN.cols)
      val empty = new NeuronVector(0)//empty
      weightsVector(img.asWeight(NN.rows, NN.cols), empty)
      img.saveas(str + '-' + i.toString)
    }
    null
  }
}

object ImageAutoEncoderTest extends Optimizable with Workspace{
	def main(args: Array[String]): Unit = {
	  //nn = new SingleLayerAutoEncoder()(20,10).create()
	  val rows = 5
	  val cols = 5
	  nn = new ImageAutoEncoder(rows, cols, 10).create()
	  val numOfSamples = 100
	  xData = new Array(numOfSamples);
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new ImageVector(rows, cols, new Uniform(-1,1)) 
	  }
	  yData = xData
	  
	  initMemory()
	  val w = getRandomWeightVector()
	  
	  var time:Long = 0
	  
	  time = System.currentTimeMillis();
	  val (obj, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj)
	  
	  nn.asInstanceOf[InstanceOfImageAutoEncoder].displayHiddenNetwork("hidden")

	}
}