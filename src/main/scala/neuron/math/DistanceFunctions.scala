package neuron.math
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.math._

abstract class DistanceFunction {
  // Please note that grad() return the partial gradient against x
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector
  def apply(x:NeuronVector, y:NeuronVector): Double
  def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix
  def applyV(x:NeuronMatrix, y:NeuronMatrix): NeuronVector
  def applyWithGradV(x:NeuronMatrix, y:NeuronMatrix): (NeuronVector, NeuronMatrix) = (applyV(x,y), grad(x,y))
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double = applyV(x,y).sum
  def applyWithGrad(x:NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = (apply(x,y), grad(x,y))
}

abstract class KernelFunction extends DistanceFunction{
  // Please note that grad() return the partial gradient against x
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector
  def apply(x:NeuronVector, y:NeuronVector): Double
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = null
  override def applyWithGradV(x:NeuronMatrix, y:NeuronMatrix) = (null, grad(x, y))
  def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double
  override def applyWithGrad(x:NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = (apply(x,y), grad(x,y))
}



object L1Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = new NeuronVector(SoftThreshold((x-y).data))
  def apply(x:NeuronVector, y:NeuronVector) = sum(abs((x-y).data))
  def grad(x:NeuronMatrix, y:NeuronMatrix)  = new NeuronMatrix(SoftThreshold((x-y).data))
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = AbsFunction(x-y).sumCol()
}

object L2Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (x - y).euclideanSqrNorm /2.0
  def grad(x:NeuronMatrix, y:NeuronMatrix)  = (x - y)
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = (x - y).euclideanSqrNormCol /2.0
}

class MahDistanceByLinTrans (val l: NeuronMatrix) extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = l TransMult l * (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (l * (x - y)).euclideanSqrNorm / 2.0
  def grad(x: NeuronMatrix, y:NeuronMatrix) = l TransMult l * (x - y)
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = (l * (x - y)).euclideanSqrNormCol / 2.0
}

class MahDistanceByMetricMat (val m: NeuronMatrix) extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = m * (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = ((x - y) dot (m * (x - y))) / 2.0
  def grad(x: NeuronMatrix, y:NeuronMatrix) = m * (x - y)
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = ((x - y) :* (m * (x - y))).sumCol / 2.0  
}

object SoftMaxDistance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector) ={
    assert(abs(y.sum - 1) < 1E-6) // y must be a probability distribution
    val x1 = new NeuronVector(exp(x.data))
    (x1/x1.sum - y)
  }
  def grad(x:NeuronMatrix, y:NeuronMatrix) ={
    assert((y.sumCol() - 1.0).euclideanSqrNorm < 1E-6)
    val x1 = new NeuronMatrix(exp(x.data))
    new NeuronMatrix(x1.data(*,::) :/ sum(x1.data(::,*)).toDenseVector) - y
  }
  def apply(x:NeuronVector, y:NeuronVector) = {
    val x1 = new NeuronVector(exp(x.data))
    val x2 = new NeuronVector(-log(x1.data / sum(x1.data)))
    y dot x2
  }
  def applyV(x:NeuronMatrix, y:NeuronMatrix) = {
    val x1 = new NeuronMatrix(exp(x.data))
    val x2 = new NeuronMatrix(-log(x1.data(*,::) :/ sum(x1.data(::,*)).toDenseVector))
    (y :* x2).sumCol()
  }
}

class BlockKernelDistance (d: KernelFunction, blocks: List[Range]) extends KernelFunction {
	def grad(x:NeuronVector, y:NeuronVector) = null // not useful
	
	def apply(x:NeuronVector, y:NeuronVector) = 0 // not useful
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  val gradmat = new NeuronMatrix(x.rows, x.cols)
	  for (block <- blocks) {
	    gradmat.Rows(block) :+= (d.grad(x.Rows(block), y.Rows(block)) :/= (block.length * block.length))
	  }
	  gradmat
	}
	
	override def apply(x: NeuronMatrix, y:NeuronMatrix): Double = {
	  val value: Double = 0
	  blocks.map(
	    block => d(x.Rows(block), y.Rows(block)) / (block.length * block.length)
	  ).reduce(_ + _)
	}
	
	override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  val gradmat = new NeuronMatrix(x.rows, x.cols)
	  var value = 0.0
	  for (block <- blocks) {
	    val tmp = d.applyWithGrad(x.Rows(block), y.Rows(block))
	    value = value + tmp._1 / (block.length * block.length)
	    gradmat.Rows(block) :+= (tmp._2 :/= (block.length * block.length))
	  }
	  (value, gradmat)
	}
		
}

class SquaredKernelDistance (mu:Double = 0.0) extends KernelFunction {
	def grad(x:NeuronVector, y:NeuronVector) = null // not useful
	
	def apply(x:NeuronVector, y:NeuronVector) = 0 // not useful
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  val xxtensor = (x TransMult x); 
	  val yxtensor = (y TransMult x); 
	  ((x * xxtensor) / (x.cols) - (y * yxtensor) / (x.cols / 2.0)) 
	}
	
	override def apply(x: NeuronMatrix, y:NeuronMatrix): Double = {
	  val xxtensor = x TransMult x; xxtensor.diagonal():=0.0
	  val yxtensor = y TransMult x; yxtensor.diagonal():=0.0
	  val yytensor = y TransMult y; yytensor.diagonal():=0.0
	  ((xxtensor :* xxtensor).sumAll + (yytensor :* yytensor).sumAll) / (2* x.cols-2) - (yxtensor :* yxtensor).sumAll / (x.cols-1)
	}
	
	// mu makes differences
   override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  val xxtensor = x TransMult x; 
	  val yxtensor = y TransMult x;
	  val yytensor = y TransMult y; 
	  if (mu != 0.0) 
	  (((xxtensor:*xxtensor).sumAll + (yytensor:*yytensor).sumAll) / (2*x.cols) - (yxtensor:*yxtensor).sumAll/x.cols +
			 ((xxtensor.diagonal().sum() + yytensor.diagonal().sum()) / (2.0) - yxtensor.diagonal().sum()) * mu,
	      (x * xxtensor- y * yxtensor) / (x.cols / 2.0) + (x - y) * mu)
	  else {
	    val grad = (x * xxtensor- y * yxtensor) / (x.cols / 2.0)
	    xxtensor.diagonal():=0.0
	    yxtensor.diagonal():=0.0
	    yytensor.diagonal():=0.0
	    val value = if (x.cols > 1) {
	      ((xxtensor:*xxtensor).sumAll + (yytensor:*yytensor).sumAll) / (2*x.cols-2) - (yxtensor:*yxtensor).sumAll/(x.cols-1) 
	    } else {
	      0.0
	    }
	    (value, grad)
	  }
	}
	    
}

class KernelDistance(kernel: NeuronFunction, mu: Double = 0.0) extends KernelFunction {
	def grad(x:NeuronVector, y:NeuronVector) = 
	  x * kernel.grad(x dot x) - y * kernel.grad(y dot x)
	
	def apply(x:NeuronVector, y:NeuronVector) = 
	  (kernel(x dot x) - 2 * kernel(x dot y) + kernel(y dot y))/2.0
	
	def grad(x:NeuronMatrix, y:NeuronMatrix) = {
	  val xxtensor = kernel.grad(x TransMult x); 
	  val yxtensor = kernel.grad(y TransMult x);
	  (x * xxtensor) / (x.cols) - (y * yxtensor) / (x.cols)
	}
	
	override def apply(x: NeuronMatrix, y:NeuronMatrix): Double = {
	  val xxtensor = kernel(x TransMult x); 
	  val yxtensor = kernel(y TransMult x);
	  val yytensor = kernel(y TransMult y); 
	  (xxtensor.sumAll + yytensor.sumAll) / (2* x.cols) - yxtensor.sumAll / (x.cols)
	}
	
	override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  val xxtensor = x TransMult x; 
	  val yxtensor = y TransMult x;
	  val yytensor = y TransMult y; 
	  ((kernel(xxtensor).sumAll + kernel(yytensor).sumAll) / (2*x.cols) - kernel(yxtensor).sumAll/x.cols +
	      ((xxtensor.diagonal().sum() + yytensor.diagonal().sum()) / (2.0) - yxtensor.diagonal().sum()) * mu,
	      (x * kernel.grad(xxtensor)- y * kernel.grad(yxtensor)) / x.cols + (x - y) * mu)
	}
	  
}

object HistogramIntersectionKernelDistance extends KernelFunction {
  def grad (x: NeuronVector, y:NeuronVector) =
    new NeuronVector(SoftThreshold(x.data-y.data)/2.0)
  def apply(x: NeuronVector, y:NeuronVector) = 
    sum(abs(x.data - y.data)) / 2.0
  def grad (x: NeuronMatrix , y: NeuronMatrix) = {
    val g = new NeuronMatrix (x.rows, x.cols)
    for (i<- 0 until x.cols) {       
       g.colVec(i) := (new NeuronMatrix(SoftThreshold(x.data(::,*) - x.data(::,i))).sumRow()) - 
       new NeuronMatrix(SoftThreshold(y.data(::,*) - x.data(::,i))).sumRow()          
    }
    g / (x.cols)
  }
  
  override def apply(x: NeuronMatrix, y: NeuronMatrix): Double = {
    (for (i<-0 until x.cols) yield {      
       sum((x.data(::, *) + x.data(::,i)) - abs(x.data(::, *) - x.data(::,i)))/2.0 + 
       sum((y.data(::,*) + y.data(::,i)) - abs(y.data(::,*) - y.data(::,i)))/2.0 - 
       sum((y.data(::,*) + x.data(::,i)) - abs(y.data(::,*) - x.data(::,i)))       
    }).sum / (x.cols)
  }
}

/** (experimental) U statistics */
class KernelDistanceU(kernel: NeuronFunction, mu: Double = 0.0) extends KernelFunction {
	def grad(x:NeuronVector, y:NeuronVector) = 
	  x * kernel.grad(x dot x) - y * kernel.grad(y dot x)
	
	def apply(x:NeuronVector, y:NeuronVector) = 
	  (kernel(x dot x) - 2 * kernel(x dot y) + kernel(y dot y))/2.0
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  if (x.cols == 1)
	    return new NeuronMatrix(x.rows, x.cols)
	  val xxtensor = kernel.grad(x TransMult x); xxtensor.diagonal():=0.0
	  val yxtensor = kernel.grad(y TransMult x); yxtensor.diagonal():=0.0
	  (x * xxtensor - y * yxtensor) / (x.cols - 1)
	}
	
  override def apply(x: NeuronMatrix, y:NeuronMatrix): Double = {
    if (x.cols == 1)
	    return 0
	  val xxtensor = kernel(x TransMult x); xxtensor.diagonal():=0.0
	  val yxtensor = kernel(y TransMult x); yxtensor.diagonal():=0.0
	  val yytensor = kernel(y TransMult y); yytensor.diagonal():=0.0
	  (xxtensor.sumAll + yytensor.sumAll -  2* yxtensor.sumAll) / (2*(x.cols -1))
	}
	
	override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  if (x.cols == 1)
	    return (0, new NeuronMatrix(x.rows, x.cols))
	  val xxtensor = x TransMult x; val xxdiag = xxtensor.diagonal().sum(); xxtensor.diagonal():= 0.0
	  val yxtensor = y TransMult x; val yxdiag = yxtensor.diagonal().sum(); yxtensor.diagonal():= 0.0
	  val yytensor = y TransMult y; val yydiag = yytensor.diagonal().sum(); yytensor.diagonal():= 0.0
	  
	  val Ustat = (kernel(xxtensor).sumAll + kernel(yytensor).sumAll - 2 * kernel(yxtensor).sumAll) / ( 2* (x.cols -1))
	  
		(Ustat, (x * kernel.grad(xxtensor)) / (x.cols -1) - (y * kernel.grad(yxtensor)) / (x.cols-1) + (x - y) * (mu / x.cols) )
	}
	  
}