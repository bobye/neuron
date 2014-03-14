package neuralnetwork

/********************************************************************************************/
// Numerical classes and their operations : interface to breeze
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._

import breeze.stats.distributions._
//import breeze.math._


class NeuronVector (var data: DenseVector[Double]) {
  val length = data.length
  def this(n:Int) = this(DenseVector.zeros[Double] (n))
  def this(n:Int, rand: Rand[Double]) = this(DenseVector.rand(n, rand)) // uniform sampling, might not be a good default choice
  def concatenate (that: NeuronVector) : NeuronVector = new NeuronVector(DenseVector.vertcat(this.data, that.data))
  def splice(num: Int) : (NeuronVector, NeuronVector) = (new NeuronVector(this.data(0 until num)), new NeuronVector(this.data(num to -1)))

  def -(that:NeuronVector): NeuronVector = new NeuronVector(this.data - that.data)
  def +(that:NeuronVector): NeuronVector = new NeuronVector(this.data + that.data)
  def *(x:Double) : NeuronVector = new NeuronVector(this.data * x)
  def /(x:Double) : NeuronVector = new NeuronVector(this.data / x)
  def :=(that: NeuronVector): Unit = {this.data := that.data; }
  def +=(that: NeuronVector): Unit = {this.data :+= that.data; }
  def :*=(x:Double): Unit = {this.data :*= x}
  def :/=(x:Double): Unit = {this.data :/= x}
  def euclideanSqrNorm = {val z = norm(data); z*z}
  def DOT(that: NeuronVector): NeuronVector = new NeuronVector(this.data :* that.data)
  def CROSS (that: NeuronVector): Weight = new Weight(this.data.asDenseMatrix.t * that.data.asDenseMatrix)
  
  def set(x:Double) : Unit = {data:=x; }
  def copy(): NeuronVector = new NeuronVector(data.copy)
  def sum(): Double = data.sum
  def asWeight(rows:Int, cols:Int): Weight = new Weight (data.asDenseMatrix.reshape(rows, cols)) 
  //override def toString() = data.toString
}
class Weight (var data:DenseMatrix[Double]){
  def this(rows:Int, cols:Int) = this(DenseMatrix.zeros[Double](rows,cols))
  def this(rows:Int, cols:Int, rand: Rand[Double]) = this(DenseMatrix.rand(rows, cols, rand)) // will be fixed in next release
  def *(x:NeuronVector):NeuronVector = new NeuronVector(data * x.data)
  def Mult(x:NeuronVector) = this * x
  def TransMult(x:NeuronVector): NeuronVector = new NeuronVector(this.data.t * x.data)
  def *(x:Double): Weight = new Weight(this.data * x)
  def :=(that:Weight): Unit = {this.data := that.data}
  def +=(that:Weight): Unit = {this.data :+= that.data}
  def :*=(x:Double): Unit = {this.data :*= x}
  def vec(isView: Boolean = true) = new NeuronVector(data.flatten(isView))  // important!
  def transpose = new Weight(data.t)
  def set(x: Double) : Unit={data:=x; }
  def euclideanSqrNorm: Double = {val z = norm(data.flatten()); z*z}
}

class WeightVector (data: DenseVector[Double]) extends NeuronVector(data) {
  def this(n:Int) = this(DenseVector.zeros[Double](n))
  def this(n:Int, rand: Rand[Double]) = this(DenseVector.rand(n, rand))
  var ptr : Int = 0
  def reset(): Unit = {ptr = 0; }
  def apply(W:Weight, b:NeuronVector): Int = {
    var rows = W.data.rows
    var cols = W.data.cols
    
    W.data = data(ptr until ptr + rows*cols).asDenseMatrix.reshape(rows, cols)
    ptr = (ptr + rows * cols) % length
    b.data = data(ptr until ptr + b.length)
    ptr = (ptr + b.length) % length
    ptr
  }
  def set(wv: NeuronVector): Int = {
    ptr = 0
    data := wv.data
    0
  }
  override def copy(): WeightVector = new WeightVector(data.copy)
}


object NullVector extends NeuronVector (0)
object OneVector extends NeuronVector(DenseVector(1.0)) 

object NullWeight extends Weight (0,0)

abstract class NeuronFunction {
  def grad(x:NeuronVector): NeuronVector
  def apply(x:NeuronVector): NeuronVector
}

object IndentityFunction extends NeuronFunction {
  def grad(x:NeuronVector): NeuronVector = new NeuronVector(DenseVector.ones[Double](x.data.length))
  def apply(x:NeuronVector) = x
}

object sigmoid extends UFunc with MappingUFunc{
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = 1/(1+scala.math.exp(-a))
    }
}
object dsgm extends UFunc with MappingUFunc{
    implicit object implDouble extends Impl [Double, Double] {
      def apply(a:Double) = {
        var b = scala.math.exp(-a)
        b/((1+b)*(1+b))
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
  def grad(x:NeuronVector): NeuronVector = new NeuronVector(dsgm(x.data)) // can be simplified, if apply() is applied first
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(sigmoid(x.data))
}

class KL_divergenceFunction(val rho: Double) extends NeuronFunction {
  object dKLdfunc extends dKLd(rho)
  object KLdiv extends KLdiv(rho)
  def grad(x:NeuronVector): NeuronVector = new NeuronVector(dKLdfunc(x.data))
  def apply(x:NeuronVector): NeuronVector = new NeuronVector(KLdiv(x.data))
}

abstract class DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector): NeuronVector
  def apply(x:NeuronVector, y:NeuronVector): Double
}

object L2Distance extends DistanceFunction {
  def grad(x:NeuronVector, y:NeuronVector) = (x - y)
  def apply(x:NeuronVector, y:NeuronVector) = (x - y).euclideanSqrNorm /2.0
}

/********************************************************************************************/
// Implement batch mode training 
abstract trait Optimizable {
  /*************************************/
  // To be specified 
  var nn: InstanceOfNeuralNetwork = null
  var xData : Array[NeuronVector] = null
  var yData : Array[NeuronVector] = null
  /*************************************/
  
  final var randomGenerator = new scala.util.Random
  
  def initMemory() : (InstanceOfNeuralNetwork, SetOfMemorables) = {
    val seed = System.currentTimeMillis().hashCode.toString
    var mem = new SetOfMemorables
    nn.init(seed, mem).allocate(seed, mem)
    (nn, mem)
  }
  
  def getRandomWeightVector (amplitude:Double = 1.0, rand:Rand[Double] = new Uniform(-1,1)) : WeightVector = {
    assert(amplitude > 0)
    
    val wdefault = nn.getWeights(System.currentTimeMillis().hashCode.toString) // get dimension of weights
    val rv = new WeightVector(wdefault.length, rand) 
    //rv :*= amplitude
    rv := wdefault
    //println(rv.data)
    //println(wdefault.data)
    rv
  }
  
  def getObj(w: WeightVector, distance:DistanceFunction = L2Distance) : Double = { // doesnot compute gradient or backpropagation
    val size = xData.length
    assert (size >= 1 && size == yData.length)
    var totalCost: Double = 0.0
    val dw = new WeightVector(w.length)
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w, dw)
    var (_, mem) = initMemory()
    for (i <- 0 until size) {
      totalCost = totalCost + distance(nn(xData(i), mem), yData(i))
    }
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString)
    totalCost/size + regCost
  }
  
  def getObjAndGrad (w: WeightVector, distance:DistanceFunction = L2Distance): (Double, NeuronVector) = {
    val size = xData.length
    assert(size >= 1 && size == yData.length)
    var totalCost:Double = 0.0
    /*
     * Compute objective and gradients in batch mode
     * which can be run in parallel 
     */
    val dw = new WeightVector(w.length)
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w, dw)
    var (_, mem) = initMemory()
    for (i <- 0 until size) { // feedforward pass
      nn(xData(i), mem)
    }
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w, dw)
    for {i <- 0 until size} {
      val x = nn(xData(i), mem); val y = yData(i)
      var z = distance.grad(x, yData(i))
      totalCost = totalCost + distance(x,y)
      nn.backpropagate(z, mem) // update dw !      
    }
    /*
     * End parallel loop
     */
    // println(totalCost/size, regCost)
    dw :/= size
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString)
    (totalCost/size + regCost, dw)
  }
  
  def getApproximateObjAndGrad (w: WeightVector, distance:DistanceFunction = L2Distance) : (Double, NeuronVector) = {
    // Compute gradient using numerical approximation
    var dW = w.copy()
    for (i<- 0 until w.length) {
	  val epsilon = 0.00001
	  val w2 = w.copy
	  w2.data(i) = w.data(i) + epsilon
	  val cost1 = getObj(w2, distance)
	  w2.data(i) = w.data(i) - epsilon
	  val cost2 = getObj(w2, distance)
	  
	  dW.data(i) = (cost1 - cost2) / (2*epsilon)
	}
    (getObj(w), dW)
  }
  

  /*
   * Train neural network using first order minimizer (L-BFGS)
   * Please NOTE there is no regularization penalty in training 
   * But it is possible to add them in the future: 
   *  (1) L1 or L2 on weights
   *  (2) sparsity parameter
   */ 
  def train(w: WeightVector, maxIter:Int = 400, distance: DistanceFunction = L2Distance): (Double, WeightVector) = {

    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val w = new WeightVector(x)
	    val (obj, grad) = getObjAndGrad(w, distance)
	    (obj, grad.data)
	  }    
    }
    
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter)
	val w2 = new WeightVector(lbfgs.minimize(f, w.data))
    (f(w2.data), w2)
  }
}