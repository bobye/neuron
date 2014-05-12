package neuron.math
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._

class SGDmTrain (momentum: Double = 0.9, lr: Double = 0.01, maxIter: Int = 500){
	
	def minimize(f: DiffFunction[DenseVector[Double]], 
				 init: DenseVector[Double]): DenseVector[Double] = {
	  val n = init.size
	  val x = DenseVector.zeros[Double](n)
	  val u = DenseVector.zeros[Double](n)
	  
	  x := init
	  for (i<- 0 until maxIter) {
	    val (fval, grad) = f.calculate(x)
	    u *= momentum
	    u -= (grad * lr)
	    x += u
	    println(i + " fval: " + fval)
	  }
	  x
	}
}
