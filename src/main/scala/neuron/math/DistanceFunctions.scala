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
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double
  def applyWithGrad(x:NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = (apply(x,y), grad(x,y))
}

object L1Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = new NeuronVector(SoftThreshold((x-y).data))
  def apply(x:NeuronVector, y:NeuronVector) = sum(abs((x-y).data))
  def grad(x:NeuronMatrix, y:NeuronMatrix)  = new NeuronMatrix(SoftThreshold((x-y).data))
  def apply(x:NeuronMatrix, y:NeuronMatrix) = sum(abs((x-y).data))
}

object L2Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (x - y).euclideanSqrNorm /2.0
  def grad(x:NeuronMatrix, y:NeuronMatrix)  = (x - y)
  def apply(x:NeuronMatrix, y:NeuronMatrix) = (x - y).euclideanSqrNorm /2.0
}

class MahDistanceByLinTrans (val l: NeuronMatrix) extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = l TransMult l * (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (l * (x - y)).euclideanSqrNorm / 2.0
  def grad(x: NeuronMatrix, y:NeuronMatrix) = l TransMult l * (x - y)
  def apply(x:NeuronMatrix, y:NeuronMatrix) = (l * (x - y)).euclideanSqrNorm / 2.0
}

class MahDistanceByMetricMat (val m: NeuronMatrix) extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector)  = m * (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = ((x - y) dot (m * (x - y))) / 2.0
  def grad(x: NeuronMatrix, y:NeuronMatrix) = m * (x - y)
  def apply(x:NeuronMatrix, y:NeuronMatrix) = ((x - y) :* (m * (x - y))).sumAll() / 2.0  
}

object SoftMaxDistance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector ={
    assert(abs(y.sum - 1) < 1E-6) // y must be a probability distribution
    val x1 = new NeuronVector(exp(x.data))
    (x1/x1.sum - y)
  }
  def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix ={
    assert((y.sumCol() - 1.0).euclideanSqrNorm < 1E-6)
    val x1 = new NeuronMatrix(exp(x.data))
    new NeuronMatrix(x1.data(*,::) :/ sum(x1.data(::,*)).toDenseVector) - y
  }
  def apply(x:NeuronVector, y:NeuronVector): Double = {
    val x1 = new NeuronVector(exp(x.data))
    val x2 = new NeuronVector(-log(x1.data / sum(x1.data)))
    y dot x2
  }
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double = {
    val x1 = new NeuronMatrix(exp(x.data))
    val x2 = new NeuronMatrix(-log(x1.data(*,::) :/ sum(x1.data(::,*)).toDenseVector))
    (y :* x2).sumAll
  }
}

class SquaredKernelDistance (mu:Double = 0.0) extends DistanceFunction {
	def grad(x:NeuronVector, y:NeuronVector) = null // not useful
	
	def apply(x:NeuronVector, y:NeuronVector) = 0 // not useful
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  val xxtensor = (x TransMult x); 
	  val yxtensor = (y TransMult x);
	  ((x * xxtensor) / (x.cols) - (y * yxtensor) / (x.cols / 2.0)) 
	}
	
	def apply(x: NeuronMatrix, y:NeuronMatrix) = {
	  val xxtensor = x TransMult x; 
	  val yxtensor = y TransMult x;
	  val yytensor = y TransMult y; 
	  ((xxtensor :* xxtensor).sumAll + (yytensor :* yytensor).sumAll) / (2* x.cols) - 
	  (yxtensor :* yxtensor).sumAll / (x.cols)
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
	  else 
	  (((xxtensor:*xxtensor).sumAll + (yytensor:*yytensor).sumAll) / (2*x.cols) - (yxtensor:*yxtensor).sumAll/x.cols,
	      (x * xxtensor- y * yxtensor) / (x.cols / 2.0) )  
	}
	    
}

class KernelDistance(kernel: NeuronFunction, mu: Double = 0.0) extends DistanceFunction {
	def grad(x:NeuronVector, y:NeuronVector) = 
	  x * kernel.grad(x dot x) - y * kernel.grad(y dot x)
	
	def apply(x:NeuronVector, y:NeuronVector) = 
	  (kernel(x dot x) - 2 * kernel(x dot y) + kernel(y dot y))/2.0
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  val xxtensor = kernel.grad(x TransMult x); 
	  val yxtensor = kernel.grad(y TransMult x);
	  (x * xxtensor) / (x.cols) - (y * yxtensor) / (x.cols)
	}
	
	def apply(x: NeuronMatrix, y:NeuronMatrix) = {
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

object HistogramIntersectionKernelDistance extends DistanceFunction {
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
  def apply(x: NeuronMatrix, y: NeuronMatrix): Double = {
    (for (i<-0 until x.cols) yield {      
       sum(min(x.data(::, *), x.data(::,i))) + sum(min(y.data(::,*), y.data(::,i))) - 
       2.0*sum(min(y.data(::,*), x.data(::,i)))       
    }).sum / (x.cols)
  }
}

/** (experimental) U statistics */
class KernelDistanceU(kernel: NeuronFunction, mu: Double = 0.0) extends DistanceFunction {
	def grad(x:NeuronVector, y:NeuronVector) = 
	  x * kernel.grad(x dot x) - y * kernel.grad(y dot x)
	
	def apply(x:NeuronVector, y:NeuronVector) = 
	  (kernel(x dot x) - 2 * kernel(x dot y) + kernel(y dot y))/2.0
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  val xxtensor = kernel.grad(x TransMult x); xxtensor.diagonal():=0.0
	  val yxtensor = kernel.grad(y TransMult x); // yxtensor.diagonal():=0.0
	  (x * xxtensor) / (x.cols -1.0) - (y * yxtensor) / (x.cols)
	}
	
	def apply(x: NeuronMatrix, y:NeuronMatrix) = {
	  val xxtensor = kernel(x TransMult x); xxtensor.diagonal():=0.0
	  val yxtensor = kernel(y TransMult x); //yxtensor.diagonal():=0.0
	  val yytensor = kernel(y TransMult y); yytensor.diagonal():=0.0
	  (xxtensor.sumAll + yytensor.sumAll) / ( 2* (x.cols -1.0)) - yxtensor.sumAll / (x.cols)
	}
	
	override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  val xxtensor = x TransMult x; val xxdiag = xxtensor.diagonal().sum(); xxtensor.diagonal():= 0.0
	  val yxtensor = y TransMult x; val yxdiag = yxtensor.diagonal().sum()
	  val yytensor = y TransMult y; val yydiag = yytensor.diagonal().sum(); yytensor.diagonal():= 0.0
	  
	  val Ustat = (kernel(xxtensor).sumAll + kernel(yytensor).sumAll) / ( 2* (x.cols -1.0)) - kernel(yxtensor).sumAll/(x.cols)
	  
	  if (Ustat > 0)
		  (Ustat +
	       ((xxdiag + yydiag) / (2.0) - yxdiag) * (mu / x.cols)
	      , (x * kernel.grad(xxtensor)) / (x.cols -1) - (y * kernel.grad(yxtensor)) / (x.cols) + (x - y) * (mu / x.cols) )
	  else
	    (((xxdiag + yydiag) / (2.0) - yxdiag) * (mu / x.cols), 
	        (x - y) * (mu / x.cols))
	}
	  
}