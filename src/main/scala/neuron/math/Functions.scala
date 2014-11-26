package neuron.math
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.math._

abstract class NeuronFunction {
  def grad(x: Double): Double
  def apply(x:Double): Double
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector 
  def grad(x:NeuronVector): NeuronVector = grad(x, NullVector)
  def apply(x:NeuronVector): NeuronVector 
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix
  def grad(x:NeuronMatrix): NeuronMatrix = grad(x, NullMatrix)
  def apply(x:NeuronMatrix): NeuronMatrix
}

object IdentityFunction extends NeuronFunction {
  def grad(x: Double) = 1.0
  def apply(x: Double) = x
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
        else 0.0
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
  def grad(x:Double) = dsgm(x)
  def apply(x:Double) = sigmoid(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = {
    if (buf == NullVector) 
      new NeuronVector(dsgm(x.data)) // can be simplified, if apply() is applied first
    else
      buf - (buf :* buf)
  }
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(sigmoid(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = {
    if (buf == NullMatrix) 
      new NeuronMatrix(dsgm(x.data))
    else
      buf - (buf :* buf)
  }
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(sigmoid(x.data))
}

object TanhFunction extends NeuronFunction {
  def grad(x:Double) = dtanh(x)
  def apply(x:Double) = tanh(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = {
    if (buf == NullVector) 
      new NeuronVector(dtanh(x.data)) // can be simplified, if apply() is applied first
    else
      (buf :* buf)*(-1)+1
  }
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(tanh(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = {
    if (buf == NullMatrix) 
      new NeuronMatrix(dtanh(x.data))
    else
      ((buf :* buf) * (-1)) +1
  }
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(tanh(x.data))
}

object ReLUFunction extends NeuronFunction {
  def grad(x:Double) = dReLU(x)
  def apply(x:Double) = ReLU(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dReLU(x.data)) 
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(ReLU(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dReLU(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(ReLU(x.data))
}

object SoftplusFunction extends NeuronFunction {
  def grad(x:Double) = dsoftplus(x)
  def apply(x:Double) = softplus(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dsoftplus(x.data)) 
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(softplus(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dsoftplus(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(softplus(x.data))
}

class PowerFunction (r: Double) extends NeuronFunction {
  def grad(x:Double) = pow(x,r)
  def apply(x:Double) = pow(x, r-1)*r
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(pow(x.data, r)) 
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(pow(x.data, r-1)*r)
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(pow(x.data, r))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(pow(x.data, r-1)*r)
}

object ExpFunction extends NeuronFunction {
  def grad(x: Double) = exp(x)
  def apply(x: Double) = exp(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(exp(x.data))
  def apply(x:NeuronVector): NeuronVector = new NeuronVector(exp(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(exp(x.data))
  def apply(x:NeuronMatrix) = new NeuronMatrix(exp(x.data))
}

object SquareFunction extends NeuronFunction {
  def grad(x:Double) = 2*x
  def apply(x:Double) = x*x
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = x * 2
  def apply(x:NeuronVector): NeuronVector= (x :* x)
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = x * 2
  def apply(x:NeuronMatrix): NeuronMatrix = (x :* x) 
}

class Square2Function(coeff:Double = 1.0) extends NeuronFunction {
  def grad(x:Double) = 2*coeff*(coeff*x+1)
  def apply(x:Double) = (coeff*x+1)*(coeff*x +1) 
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = x * (2 * coeff*coeff) +2*coeff
  def apply(x:NeuronVector): NeuronVector= {
    val y = x * coeff + 1
    (y :* y)
  }
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = x * (2 * coeff*coeff) +2*coeff
  def apply(x:NeuronMatrix): NeuronMatrix = {
    val y = x * coeff + 1
    (y :* y)   
  }
}

object CubicFunction extends NeuronFunction {
  def grad(x:Double) = 3*x*x
  def apply(x:Double) = x* x* x
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector((x :* x).data) * 3
  def apply(x:NeuronVector): NeuronVector= (x :* x :* x)
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix((x :* x).data) * 3
  def apply(x:NeuronMatrix): NeuronMatrix = (x :* x :* x)
}

object AbsFunction extends NeuronFunction {
  def grad(x:Double) = signum(x)
  def apply(x:Double)= abs(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(signum(x.data))
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(abs(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(signum(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(abs(x.data))
}

class KL_divergenceFunction(val rho: Double) extends NeuronFunction {
  object dKLdfunc extends dKLd(rho)
  object KLdiv extends KLdiv(rho)
  def grad(x:Double)=dKLdfunc(x)
  def apply(x:Double)=KLdiv(x)
  def grad(x:NeuronVector, buf:NeuronVector): NeuronVector = new NeuronVector(dKLdfunc(x.data))
  def apply(x:NeuronVector): NeuronVector = new NeuronVector(KLdiv(x.data))
  def grad(x:NeuronMatrix, buf:NeuronMatrix): NeuronMatrix = new NeuronMatrix(dKLdfunc(x.data))
  def apply(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(KLdiv(x.data))
}

