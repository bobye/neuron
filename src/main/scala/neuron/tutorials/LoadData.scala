package neuron.tutorials
import neuron.math._
import breeze.linalg._

object LoadData {
  def rawImages64(): Array[NeuronVector] = {
      val numOfPixels = 8*8	  
	  val source = scala.io.Source.fromFile("data/UFLDL/sparseae/patches64x10000.txt")
	  val dataSource = source.getLines.toArray
	  val numOfSamples = dataSource.length
	  val xData = new Array[NeuronVector](numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(
	        new DenseVector(dataSource(i).split("\\s+").map(_.toDouble), 0, 1, numOfPixels))
	  }
      source.close()
      xData
  }
  
  def rawImages64M(): NeuronMatrix = {
      val numOfPixels = 8*8	  
	  val source = scala.io.Source.fromFile("data/UFLDL/sparseae/patches64x10000.txt")
	  val dataBlock = source.mkString.split("\\s+").map(_.toDouble)
	  source.close()
	  new NeuronMatrix(new DenseMatrix(numOfPixels, dataBlock.length/numOfPixels, dataBlock))
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
  
  def mnistTrainM(): NeuronMatrix = {
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/UFLDL/sparseae/mnist/train-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    
    val buf = new Array[Byte](numOfPixels * numOfSamples)
    source.read(buf)
    val dataBlock = buf.map(b => ((0xff & b).toDouble / 255.00) * 0.8 + 0.1) // normalized to [0.1, 0.9]
    source.close()
    new NeuronMatrix(new DenseMatrix(numOfPixels, numOfSamples, dataBlock))
  }
}