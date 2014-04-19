package tutorials
import neuralnetwork._
import breeze.linalg._

object LoadData {
  def rawImages64(): Array[NeuronVector] = {
      val rows = 8
	  val cols = 8
	  
	  val dataSource = scala.io.Source.fromFile("data/UFLDL/sparseae/patches64x10000.txt").getLines.toArray
	  val numOfSamples = dataSource.length
	  val xData = new Array[NeuronVector](numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(
	        new DenseVector(dataSource(i).split("\\s+").map(_.toDouble), 0, 1, rows*cols))
	  }
      xData
  }
  
  def mnistTrain(): Array[NeuronVector] = {
        // Load MNIST data
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/UFLDL/sparseae/mnist/train-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    
    val xData = new Array[NeuronVector](numOfSamples)
    for (i<- 0 until numOfSamples) {
      val buf = new Array[Byte](numOfPixels)
      source.read(buf)
      xData(i) = new NeuronVector(new DenseVector(buf.map(b => (0xff & b).toDouble/255.00), 0, 1, numOfPixels))
    }
    source.close()
    xData
  }
}