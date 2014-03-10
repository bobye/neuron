package tutorials

//import breeze.plot._
import breeze.linalg._
import breeze.stats.distributions._
import neuralnetwork._



// create custom Image AutoEncoder from SparseSingleLayerAE
class ImageAutoEncoder (val rows:Int, val cols:Int, override val hiddenDimension: Int) 
	extends SparseSingleLayerAE (3.0, .0001) (rows*cols, hiddenDimension){
  type Instance <: InstanceOfImageAutoEncoder
  override def create() = new InstanceOfImageAutoEncoder(this)
}
class InstanceOfImageAutoEncoder (override val NN: ImageAutoEncoder) 
	extends InstanceOfAutoEncoder(NN) //There is no InstanceOfSparseSingleLayerAE
{ 
  type Structure <: ImageAutoEncoder
  def displayHiddenNetwork (str:String) : Unit = { 
    val weightsVector = new WeightVector((NN.rows*NN.cols+1)*NN.hiddenDimension)
    val raw = inputLayer.getWeights((System.currentTimeMillis()%100000).toString) // load in optimized weights
    weightsVector := raw.asWeight(NN.hiddenDimension, NN.rows*NN.cols+1).transpose.vec
        
    for (i<- 0 until NN.hiddenDimension) { // display by hidden nodes
      val img = new NeuronVector(NN.rows*NN.cols)
      val b = new NeuronVector(1)//
      weightsVector(img.asWeight(NN.rows, NN.cols), b)
      println((img.data/norm(img.data)).data.mkString("\t")) // Just print
    }
  }
}

object ImageAutoEncoderTest extends Optimizable with Workspace{
	def main(args: Array[String]): Unit = {
	  val rows = 8
	  val cols = 8
	  val hidden = 25
	  
	  val dataSource = scala.io.Source.fromFile("data/UFLDL/sparseae/patches64x10000.txt").getLines.toArray
	  val numOfSamples = dataSource.length
	  xData = new Array(numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(
	        new DenseVector(dataSource(i).split("\\s+").map(_.toDouble), 0, 1, rows*cols))
	  }
	 
	  yData = xData
	  val top = new SingleLayerNeuralNetwork(rows*cols) 
	  val imcode = new ImageAutoEncoder(rows, cols, hidden)
	  val nnnet = (top TIMES imcode).create()
	  nn = nnnet // the same

	  
	  initMemory()
	  val amplitude = scala.math.sqrt(6.0/(rows*cols + hidden + 1))
	  val w = getRandomWeightVector(amplitude)
	  //val w = getRandomWeightVector()
	  
	  
	  var time:Long = 0
	  
	  time = System.currentTimeMillis();
	  val (obj, w2) = train(w)
	  println(System.currentTimeMillis() - time, obj)
	  
	  nnnet.secondInstance
	  	.asInstanceOf[InstanceOfImageAutoEncoder]
	  	.displayHiddenNetwork("hidden")

	}
}