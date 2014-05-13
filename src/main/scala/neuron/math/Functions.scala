package neuron.math
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.math._

abstract class NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector 
  def grad(x:NeuronVector): NeuronVector = grad(x, NullVector)
  def apply(x:NeuronVector): NeuronVector 
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix
  def grad(x:NeuronMatrix): NeuronMatrix = grad(x, NullMatrix)
  def apply(x:NeuronMatrix): NeuronMatrix
}

object IdentityFunction extends NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new OnesVector(x.length)
  def apply(x:NeuronVector) = x
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new OnesMatrix(x.rows, x.cols)
  def apply(x:NeuronMatrix) = x
}

object inverse extends UFunc with MappingUFunc {
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = 1.0/a
    }  
}
object SoftThreshold extends UFunc with MappingUFunc {
  val thres = 1E-3
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        if (a > thres) 1.0
        else if (a < -thres) -1.0
        else 0
      }
    }   
}
/* sigmoid is included in breeze.numerics
 * 
object sigmoid extends UFunc with MappingUFunc{
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = 1/(1+scala.math.exp(-a))
    }
}
*/
object dsgm extends UFunc with MappingUFunc{ //computing gradient of sigmoid is expensive
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        var b = scala.math.exp(-a)
        b/((1+b)*(1+b))
      }
    }
}
/* tanh is included in breeze.numerics
 * 
object tanh extends UFunc with MappingUFunc{
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = tanh(-a))
    }
}
*/
object dtanh extends UFunc with MappingUFunc{
  implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        val x = tanh(a) // [-1,1]
        1 - x*x
      }
  }
}
object ReLU extends UFunc with MappingUFunc{
  implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        if (a<0) 0 else a
      }
  }
}
object dReLU extends UFunc with MappingUFunc{
  implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        if (a<0) 0 else 1
      }
  }
}

object softplus extends UFunc with MappingUFunc{
  implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        log(1+exp(a))
      }
  }
}
object dsoftplus extends UFunc with MappingUFunc{
  implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        1/(1+exp(-a))
      }
  }
}


class KLdiv (rho:Double) extends UFunc with MappingUFunc {
  implicit object implDouble extends Impl [Double, Double] {
    def apply(a: Double) = {
      rho * log(rho/a) + (1-rho) * log((1-rho)/(1-a))
    }
  }
}
class dKLd (rho:Double) extends UFunc with MappingUFunc {
  implicit object implDouble extends Impl [Double, Double] {
    def apply(a: Double) = {
      - rho/a + (1-rho)/(1-a)
    }
  }
}
  
object SigmoidFunction extends NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = {
    if (buf == NullVector) 
      new NeuronVector(dsgm(x.data)) // can be simplified, if apply() is applied first
    else
      buf - (buf DOT buf)
  }
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(sigmoid(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = {
    if (buf == NullMatrix) 
      new NeuronMatrix(dsgm(x.data))
    else
      buf - (buf DOT buf)
  }
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(sigmoid(x.data))
}

object TanhFunction extends NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = {
    if (buf == NullVector) 
      new NeuronVector(dtanh(x.data)) // can be simplified, if apply() is applied first
    else
      (buf DOT buf)*(-1)+1
  }
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(tanh(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = {
    if (buf == NullMatrix) 
      new NeuronMatrix(dtanh(x.data))
    else
      ((buf DOT buf) Mult (-1)) +1
  }
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(tanh(x.data))
}

object ReLUFunction extends NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dReLU(x.data)) 
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(ReLU(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dReLU(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(ReLU(x.data))
}

object SoftplusFunction extends NeuronFunction {
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dsoftplus(x.data)) 
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(softplus(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dsoftplus(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(softplus(x.data))
}

class KL_divergenceFunction(val rho: Double) extends NeuronFunction {
  object dKLdfunc extends dKLd(rho)
  object KLdiv extends KLdiv(rho)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dKLdfunc(x.data))
  def apply(x:NeuronVector): NeuronVector = new NeuronVector(KLdiv(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dKLdfunc(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(KLdiv(x.data))
}

abstract class DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector
  def apply(x:NeuronVector, y:NeuronVector): Double
  def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double
}

object L1Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector) = new NeuronVector(SoftThreshold((x-y).data))
  def apply(x:NeuronVector, y:NeuronVector) = sum(abs((x-y).data))
  def grad(x:NeuronMatrix, y:NeuronMatrix) = new NeuronMatrix(SoftThreshold((x-y).data))
  def apply(x:NeuronMatrix, y:NeuronMatrix) = sum(abs((x-y).data))
}

object L2Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector = (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (x - y).euclideanSqrNorm /2.0
  def grad(x:NeuronMatrix, y:NeuronMatrix): NeuronMatrix = (x - y)
  def apply(x:NeuronMatrix, y:NeuronMatrix) = (x - y).euclideanSqrNorm /2.0
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
    val x1SumInv = new NeuronVector(inverse(x1.sumCol().data))
    (x1 MultElemTrans x1SumInv) - y
  }
  def apply(x:NeuronVector, y:NeuronVector): Double = {
    val x1 = new NeuronVector(exp(x.data))
    val x2 = new NeuronVector(-log(x1.data / x1.data.sum))
    (y DOT x2).sum
  }
  def apply(x:NeuronMatrix, y:NeuronMatrix): Double = {
    val x1 = new NeuronMatrix(exp(x.data))
    val x1SumInv = new NeuronVector(inverse(x1.sumCol().data))
    (y DOT (x1 MultElemTrans x1SumInv)).sumAll
  }
}