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
  
  def rawImages64V(): Array[DenseVector[Double]] = {
      val numOfPixels = 8*8	  
	  val source = scala.io.Source.fromFile("data/UFLDL/sparseae/patches64x10000.txt")
	  val dataSource = source.getLines.toArray
	  val numOfSamples = dataSource.length
	  val xData = new Array[DenseVector[Double]](numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = 
	        new DenseVector(dataSource(i).split("\\s+").map(_.toDouble), 0, 1, numOfPixels)
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
  
  def mnistImages(): Array[NeuronVector] = {
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
  
  def mnistImagesV(): Array[DenseVector[Double]] = {
        // Load MNIST data
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/UFLDL/sparseae/mnist/train-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    
    val xData = new Array[DenseVector[Double]](numOfSamples)
    for (i<- 0 until numOfSamples) {
      val buf = new Array[Byte](numOfPixels)
      source.read(buf)
      xData(i) = new DenseVector(buf.map(b => (0xff & b).toDouble/255.00), 0, 1, numOfPixels)
    }
    source.close()
    xData
  }  
  
  def unique[A](ls: List[A]) = {
  def loop(set: Set[A], ls: List[A]): List[A] = ls match {
    case hd :: tail if set contains hd => loop(set, tail)
    case hd :: tail => hd :: loop(set + hd, tail)
    case Nil => Nil
  }

  loop(Set(), ls)
  }
  
  def mnistDataM(dataName:String = "train"): (NeuronMatrix, NeuronMatrix) = {
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/UFLDL/sparseae/mnist/" + dataName + "-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    
    val buf = new Array[Byte](numOfPixels * numOfSamples)
    source.read(buf)
    val dataBlock = buf.map(b => ((0xff & b).toDouble / 255.00)) 
    //val dataBlock = buf.map(b => ((0xff & b).toDouble / 255.00) *0.8 + 0.1) // normalized to [0.1, 0.9]: improve convergence and prevent dead unit
    source.close()
    
    
    val source2 = new DataInputStream(new FileInputStream("data/UFLDL/sparseae/mnist/" + dataName + "-labels.idx1-ubyte"))
    println("magic: " + source2.readInt())
    val numOfSamples2 = source2.readInt(); println("numOfImages: " + numOfSamples2)
    val buf2 = new Array[Byte](numOfSamples2)
    source2.read(buf2)
    val dataBlock2 = buf2.map(_.toInt)
    val numOfLabels = unique(dataBlock2.toList).length
    source2.close()
    
    val labelMat = new NeuronMatrix(numOfLabels, numOfSamples2)
    (0 until numOfSamples).map(i=> {
      labelMat.data(dataBlock2(i), i) = 1
    })    
    println("Finish Loading!")
    (new NeuronMatrix(new DenseMatrix(numOfPixels, numOfSamples, dataBlock)), labelMat)
  }
  
  
  /*
  def mnistDataM_rotate(dataName:String = "train", numOfSamples: Int = 12000): (NeuronMatrix, NeuronMatrix) = {
      import java.io._
    import java.util.Scanner
      val numOfPixels: Int = 28*28	 
      val numOfLabels = 10
	  val source = new Scanner(new FileInputStream("data/UFLDL/sparseae/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_" + dataName + ".amat"))
	  val dataBlock: Array[Double] = new Array[Double](numOfSamples* (numOfPixels+1))
	  for (i<- 0 until dataBlock.length) {dataBlock(i) = source.nextDouble()}
      source.close()
	  val data = new NeuronMatrix(new DenseMatrix(numOfPixels+1, numOfSamples, dataBlock))
      val dataLabel = data.rowVec(numOfPixels).data.toArray.map(_.toInt)
      
      val dataMat = data.Rows(0 until numOfPixels)
      val labelMat = new NeuronMatrix(numOfLabels, numOfSamples)
      (0 until numOfSamples).map(i=> {
    	  labelMat.data(dataLabel(i), i) = 1
      })     
      
      val os = new DataOutputStream (new FileOutputStream("data/UFLDL/sparseae/mnist/" + dataName + "_rotate_rbimg-images.idx3-ubyte"))
      os.writeInt(1111) // magic number
      os.writeInt(numOfSamples)
      os.writeInt(28)
      os.writeInt(28)
      (0 until dataMat.cols).map(i=> {
          os.write((data.colVec(i).data*255.0).data.map(_.toInt.toByte), 0, dataMat.rows)
      })
      os.close()
      
      val os2 = new DataOutputStream (new FileOutputStream("data/UFLDL/sparseae/mnist/" + dataName + "_rotate_rbimg-labels.idx1-ubyte"))
      os2.writeInt(1111) // magic number
      os2.writeInt(numOfSamples)
      (0 until dataMat.cols).map(i=> {
          os2.writeByte(dataLabel(i).toByte)
      })
      os2.close()      

      
      (dataMat, labelMat)
  }  
  */
}
