package neuron.examples
import neuron.math._
import breeze.linalg._

object LoadData {
  
  def rawImages64(): Array[NeuronVector] = {
      val numOfPixels = 8*8	  
	  val source = scala.io.Source.fromFile("data/UFLDL/patches64x10000.txt")
	  val dataSource = source.getLines.toArray
	  val numOfSamples = dataSource.length
	  val xData = new Array[NeuronVector](numOfSamples)
	  for (i<- 0 until numOfSamples) {
	    xData(i) = new NeuronVector(dataSource(i).split("\\s+").map(_.toDouble))
	  }
      source.close()
      xData
  }
  
  def rawImages64V(): Array[DenseVector[Double]] = {
      val numOfPixels = 8*8	  
	  val source = scala.io.Source.fromFile("data/UFLDL/patches64x10000.txt")
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
	  val source = scala.io.Source.fromFile("data/UFLDL/patches64x10000.txt")
	  val dataBlock = source.mkString.split("\\s+").map(_.toDouble)
	  source.close()
	  new NeuronMatrix(numOfPixels, dataBlock.length/numOfPixels, dataBlock)
  }
  def colorPalette(): (Array[NeuronVector], Array[NeuronVector]) = {
    val dataSource = scala.io.Source.fromFile("data/colorpalette/mturkData.txt").getLines.toArray
    val labelSource= scala.io.Source.fromFile("data/colorpalette/mturkData-s.txt").getLines.toArray
    val numOfSamples = dataSource.length
    val xData = new Array[NeuronVector](numOfSamples)
    val yData = new Array[NeuronVector](numOfSamples)

    for (i<- 0 until numOfSamples) {
      xData(i) = new NeuronVector(dataSource(i).split("\\s+").map(_.toDouble))
      yData(i) = new NeuronVector(labelSource(i).split("\\s+").map(_.toDouble))
    }       
    (xData, yData)
  }
  def colorPaletteM(): (NeuronMatrix, NeuronMatrix) = {
    val dataSource = scala.io.Source.fromFile("data/colorpalette/mturkData.txt")
    val labelSource= scala.io.Source.fromFile("data/colorpalette/mturkData-s.txt")
     
    val dataBlock = dataSource.mkString.split("\\s+").map(_.toDouble)
    val labelBlock = labelSource.mkString.split("\\s+").map(_.toDouble)
    dataSource.close()
    labelSource.close()

    val rows = 15
    val cols = dataBlock.length / rows
    (new NeuronMatrix(rows, cols, dataBlock), new NeuronMatrix(1, cols, labelBlock))
  }
  def mnistImages(): Array[NeuronVector] = {
        // Load MNIST data
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/mnist/ubyte/std/train-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    
    val xData = new Array[NeuronVector](numOfSamples)
    for (i<- 0 until numOfSamples) {
      val buf = new Array[Byte](numOfPixels)
      source.read(buf)
      xData(i) = new NeuronVector(buf.map(b => (0xff & b).toDouble/255.00))
    }
    source.close()
    xData
  }
  
  def mnistImagesV(): Array[DenseVector[Double]] = {
        // Load MNIST data
    import java.io._
    val source = new DataInputStream(new FileInputStream("data/mnist/ubyte/std/train-images.idx3-ubyte"))
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
  
  def mnistDataM(dataName:String = "std", part:String = "train", 
      isLabelBinary: Boolean = true, outlierRatio: Double = 0.0): (NeuronMatrix, NeuronMatrix) = {
    import java.io._
    assert(outlierRatio <= 0.9)
    val source = new DataInputStream(new FileInputStream("data/mnist/ubyte/" + dataName + "/" + part + "-images.idx3-ubyte"))
    println("magic: " + source.readInt())
    val numOfSamples = source.readInt(); println("numOfImages: " + numOfSamples)
    val numOfOutliers = (numOfSamples * outlierRatio / (1-outlierRatio)).toInt; println("numOfOutliers: " + numOfOutliers)
    val numOfRows = source.readInt(); println("numOfRows: " + numOfRows)
    val numOfCols = source.readInt(); println("numOfCols: " + numOfCols)
    val numOfPixels = numOfRows * numOfCols
    val buf = new Array[Byte](numOfPixels * (numOfSamples + numOfOutliers))
    
    if (numOfOutliers != 0) {
    val sourceOutliers = new DataInputStream(new FileInputStream("data/background/images/train-images.idx3-ubyte"))
    sourceOutliers.readInt();sourceOutliers.readInt();sourceOutliers.readInt();sourceOutliers.readInt();        
    sourceOutliers.read(buf)
    sourceOutliers.close()
    }
    source.read(buf)

    val dataBlock = buf.map(b => ((0xff & b).toDouble / 255.00)) 
    source.close()
    
    
    
    val source2 = new DataInputStream(new FileInputStream("data/mnist/ubyte/" + dataName + "/" + part + "-labels.idx1-ubyte"))
    println("magic: " + source2.readInt())
    val numOfSamples2 = source2.readInt(); println("numOfImages: " + numOfSamples2)
    val buf2 = new Array[Byte](numOfSamples2)
    source2.read(buf2)
    val dataBlock2 = buf2.map(_.toInt)
    val numOfLabels = unique(dataBlock2.toList).length
    source2.close()
    
    if (isLabelBinary) {
    	val labelMat = new NeuronMatrix(numOfLabels, numOfSamples2)
    	(0 until numOfSamples).map(i=> {
    		labelMat.data(dataBlock2(i), i) = 1
    	})    
    	println("Finish Loading!")
    	(new NeuronMatrix(numOfPixels, numOfSamples, dataBlock), labelMat)
    } else {
      val labelMat = new NeuronMatrix(1, numOfSamples2)
      (0 until numOfSamples).map(i=> {
    		labelMat.data(0, i) = dataBlock2(i).toDouble
    	})
    	(new NeuronMatrix(numOfPixels, numOfSamples + numOfOutliers, dataBlock), labelMat)
    }
  }
}