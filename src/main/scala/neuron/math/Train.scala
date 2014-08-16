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
	  
	  var stat = 0.0
	  val statCount = 100
	  var lowest = 1E10
	  x := init
	  var time = System.currentTimeMillis()
	  for (i<- 0 until maxIter) {
	    val (fval, grad) = f.calculate(x)
	    stat = stat + fval
	    u *= momentum
	    u -= (grad * lr)
	    x += u
	    if ((i+1) % statCount == 0) {
	      if ((stat/statCount) < lowest) lowest = (stat/statCount)
	      println((i+1) + " fsum: " + stat/statCount + " fmin " + lowest, System.currentTimeMillis() - time)
	      stat = 0.0
	      time = System.currentTimeMillis()
	    }
	  }
	  x
	}
}
