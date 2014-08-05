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

class KernelDistance(kernel: NeuronFunction) extends DistanceFunction {
	def grad(x:NeuronVector, y:NeuronVector) = 
	  x * kernel.grad(x dot x) - y * kernel.grad(y dot x)
	
	def apply(x:NeuronVector, y:NeuronVector) = 
	  (kernel(x dot x) - 2 * kernel(x dot y) + kernel(y dot y))/2.0
	
	def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = {
	  (x * kernel.grad(x TransMult x) - y * kernel.grad(y TransMult x)) / (x.cols )
	}
	
	def apply(x: NeuronMatrix, y:NeuronMatrix) = {
	  (kernel(x TransMult x).sumAll - 2 * kernel(y TransMult x).sumAll + kernel(y TransMult y).sumAll)/ (2.0 * x.cols)
	}
	
	override def applyWithGrad(x: NeuronMatrix, y:NeuronMatrix): (Double, NeuronMatrix) = {
	  val xxtensor = x TransMult x
	  val yxtensor = y TransMult x
	  val yytensor = y TransMult y
	  ((kernel(xxtensor).sumAll - 2.0 * kernel(yxtensor).sumAll + kernel(yytensor).sumAll)/(2.0 * x.cols),
	      (x * kernel.grad(xxtensor) - y * kernel.grad(yxtensor))/x.cols )
	}
	  
}