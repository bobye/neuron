// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updates: Mar 2014
// Copyright under MIT License
package neuralnetwork

/********************************************************************************************/
// Numerical classes and their operations 
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._

import breeze.stats.distributions._
//import breeze.math._

class NeuronVector (val data: DenseVector[Double]) {
  val length = data.length
  def this(n:Int) = this(DenseVector.zeros[Double] (n))
  def this(n:Int, rand: Rand[Double] =  new Uniform(-1,1)) = this(DenseVector.rand(n, rand)) // uniform sampling, might not be a good default choice
  def concatenate (that: NeuronVector) : NeuronVector = new NeuronVector(DenseVector.vertcat(this.data, that.data))
  def splice(num: Int) : (NeuronVector, NeuronVector) = (new NeuronVector(this.data(0 until num)), new NeuronVector(this.data(num to -1)))

  def -(that:NeuronVector): NeuronVector = new NeuronVector(this.data - that.data)
  def +(that:NeuronVector): NeuronVector = new NeuronVector(this.data + that.data)
  def *(x:Double) : NeuronVector = new NeuronVector(this.data * x)
  def /(x:Double) : NeuronVector = new NeuronVector(this.data / x)
  def :=(that: NeuronVector): Unit = {this.data := that.data; }
  def +=(that: NeuronVector): Unit = {this.data :+= that.data; }
  def euclideanSqrNorm = {var z = norm(data); z*z}
  def DOT(that: NeuronVector): NeuronVector = new NeuronVector(this.data :* that.data)
  def CROSS (that: NeuronVector): Weight = new Weight(this.data.asDenseMatrix.t * that.data.asDenseMatrix)
  
  def set(x:Double) : Unit = {data:=x; }
  def copy(): NeuronVector = new NeuronVector(data.copy)
  def sum(): Double = data.sum
  def asWeight(rows:Int, cols:Int): Weight = new Weight (data.asDenseMatrix.reshape(rows, cols)) 
  //override def toString() = data.toString
}
class Weight (val data:DenseMatrix[Double]){
  def this(rows:Int, cols:Int) = this(DenseMatrix.zeros[Double](rows,cols))
  //def this(rows:Int, cols:Int, rand: Rand[Double]) = this(DenseMatrix.rand(rows, cols, rand)) // will be fixed in next release
  def *(x:NeuronVector):NeuronVector = new NeuronVector(data * x.data)
  def Mult(x:NeuronVector) = this * x
  def TransMult(x:NeuronVector): NeuronVector = new NeuronVector(this.data.t * x.data)
  def +=(that:Weight): Unit = {
    this.data :+= that.data
  }
  def vec = new NeuronVector(data.toDenseVector) // make copy
  def set(x: Double) : Unit={data:=x; }
}

class WeightVector (override val data: DenseVector[Double]) extends NeuronVector(data) {
  def this(n:Int) = this(DenseVector.zeros[Double](n))
  def this(n:Int, rand: Rand[Double]) = this(DenseVector.rand(n, rand))
  var ptr : Int = 0
  def reset(): Unit = {ptr = 0; }
  def apply(W:Weight, b:NeuronVector): Int = {
    var rows = W.data.rows
    var cols = W.data.cols
    
    W.data := data(ptr until ptr + rows*cols).asDenseMatrix.reshape(rows, cols)
    ptr = (ptr + rows * cols) % length
    b.data := data(ptr until ptr + b.length)
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

/********************************************************************************************/
// Graph data structure
abstract trait DirectedGraph
abstract trait AcyclicDirectedGraph extends DirectedGraph
abstract trait BinaryTree extends AcyclicDirectedGraph
// more to added



/********************************************************************************************/
// Highest level classes for Neural Network
abstract trait Workspace{// 
  implicit class Helper[T1<:Operationable](x:T1) { 
    // Two basic operations to support combination 
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
  } 
}
/** Operationable is a generic trait that supports operations **/
abstract trait Operationable extends Workspace {
  def inputDimension:Int
  def outputDimension:Int

  def create(): InstanceOfNeuralNetwork
  def toStringGeneric(): String = this.hashCode().toString +
  	 "[" + inputDimension + "," + outputDimension + "]";
}
/** Memorable NN is instance that keep internal buffers **/
abstract trait Memorable extends InstanceOfNeuralNetwork {
  var numOfMirrors:Int = 0
  //type arrayOfData[T<:NeuronVector] = Array[T]
}

/** Implement batch mode training **/
abstract trait Optimizable {
  /*************************************/
  // To be specified 
  var nn: InstanceOfNeuralNetwork = null
  var xData : Array[NeuronVector] = null
  var yData : Array[NeuronVector] = null
  /*************************************/
  
  final var randomGenerator = new scala.util.Random
  
  def initMemory() : InstanceOfNeuralNetwork = {
    val seed = System.currentTimeMillis().hashCode.toString
    nn.init(seed).allocate(seed)
  }
  
  def getRandomWeightVector (rand: Rand[Double] =  new Uniform(-1,1)) : WeightVector = {
    val wlength = nn.getWeights(System.currentTimeMillis().hashCode.toString).length // get dimension of weights
    new WeightVector(wlength, rand)
  }
  
  def getObj(w: WeightVector) : Double = { // doesnot compute gradient or backpropagation
    val size = xData.length
    assert (size >= 1 && size == yData.length)
    var totalCost: Double = 0.0
    val regCost = nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    for (i <- 0 until size) {
      totalCost = totalCost + (nn(xData(i))-yData(i)).euclideanSqrNorm/2.0
    }
    totalCost/size + regCost
  }
  
  def getObjAndGrad (w: WeightVector): (Double, NeuronVector) = {
    val size = xData.length
    assert(size >= 1 && size == yData.length)
    var totalCost:Double = 0.0
    val dW = new NeuronVector (w.length)
    /*
     * Compute objective and gradients in batch mode
     * which can be run in parallel 
     */
    val regCost = nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    
    for {i <- 0 until size} {
      var z = nn(xData(i)) - yData(i)
      totalCost = totalCost + z.euclideanSqrNorm/2.0
      nn.backpropagate(z)
      dW += nn.getDerativeOfWeights((((i+1)*System.currentTimeMillis())%1000000).toString)
    }
    /*
     * End parallel loop
     */
    (totalCost/size + regCost, dW/size)
  }
  
  def getApproximateObjAndGrad (w: WeightVector) : (Double, NeuronVector) = {
    // Compute gradient using numerical approximation
    var dW = w.copy()
    for (i<- 0 until w.length) {
	  val epsilon = 0.00001
	  val w2 = w.copy
	  w2.data(i) = w.data(i) + epsilon
	  val cost1 = getObj(w2)
	  w2.data(i) = w.data(i) - epsilon
	  val cost2 = getObj(w2)
	  
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
  def train(w: WeightVector): (Double, WeightVector) = {
    getObj(w) // one more time
    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val w = new WeightVector(x)
	    // getObj(w) // a forward pass on all the training samples, maybe slow
	    val (obj, grad) = getObjAndGrad(w)
	    (obj, grad.data)
	  }    
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter=100, m=3)
	val w2 = new WeightVector(lbfgs.minimize(f, w.data))
    (f(w2.data), w2)
  }
}

/** Class for template of neural network **/
abstract class NeuralNetwork (val inputDimension:Int, val outputDimension:Int) extends Operationable{
  type InstanceType <: InstanceOfNeuralNetwork
  def create() : InstanceOfNeuralNetwork 
  override def toString() = "?" + toStringGeneric
}

/** Class for instance of neural network, which can be applied and trained **/
abstract class InstanceOfNeuralNetwork (val NN: Operationable) extends Operationable {
  type StructureType <: Operationable
  // basic topological structure
  def inputDimension = NN.inputDimension
  def outputDimension= NN.outputDimension
  def create() = this // self reference
  def apply (x: NeuronVector) : NeuronVector
  
  // structure to vectorization functions
  var status:String = ""

  def setWeights(seed:String, w:WeightVector) : Double // return regularization term
  def getWeights(seed:String) : NeuronVector
  def getDerativeOfWeights(seed:String) : NeuronVector
  
  // dynamic operations
  def init(seed:String) : InstanceOfNeuralNetwork = {this} // default: do nothing
  def allocate(seed:String) : InstanceOfNeuralNetwork = {this} // 
  def backpropagate(eta: NeuronVector): NeuronVector
  
  // display
  override def toString() = "#" + toStringGeneric
}

abstract class SelfTransform (val dimension: Int) extends NeuralNetwork(dimension, dimension) 
abstract class InstanceOfSelfTransform (override val NN: SelfTransform) extends InstanceOfNeuralNetwork (NN)




// basic operation to derive hierarchy structures
abstract class MergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable] 
		(val first:Type1, val second:Type2) extends Operationable{ 
}

abstract class InstanceOfMergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable]
		(override val NN: MergedNeuralNetwork[Type1, Type2]) 
		extends InstanceOfNeuralNetwork(NN) {
  val firstInstance = NN.first.create()
  val secondInstance = NN.second.create()
  
  
  override def setWeights(seed:String, w: WeightVector) : Double = {
    if (status != seed) {
      status = seed
      firstInstance.setWeights(seed, w) + 
      secondInstance.setWeights(seed, w)
    } else {
      0.0
    }
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      firstInstance.getWeights(seed) concatenate secondInstance.getWeights(seed)
    }else {
      NullVector
    }
  }
  def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      firstInstance.getDerativeOfWeights(seed) concatenate secondInstance.getDerativeOfWeights(seed)
    } else {
      NullVector
    }
  }
  override def init(seed:String) = {
    firstInstance.init(seed)
    secondInstance.init(seed)
    this // do nothing
  }
  override def allocate(seed:String) = {
    firstInstance.allocate(seed)
    secondInstance.allocate(seed)
    this
  }
  
}
class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1,Type2](first,second) {
  type Instance = InstanceOfJointNeuralNetwork[Type1,Type2]
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension 
  def create(): InstanceOfJointNeuralNetwork[Type1, Type2] = new InstanceOfJointNeuralNetwork(this)
  override def toString() = "(" + first.toString + " + " + second.toString + ")"
}

class InstanceOfJointNeuralNetwork[Type1 <: Operationable, Type2 <:Operationable]
		(override val NN: JointNeuralNetwork [Type1, Type2]) 
	extends InstanceOfMergedNeuralNetwork [Type1, Type2](NN) {
  
  type StructureType = JointNeuralNetwork[Type1, Type2]
  
  def apply (x: NeuronVector)  = {
    var (first, second) = x.splice(NN.first.inputDimension)
    firstInstance(first) concatenate secondInstance(second)
  }

  def backpropagate(eta: NeuronVector) = {
    var (firstEta, secondEta) = eta.splice(NN.first.inputDimension)
    firstInstance.backpropagate(firstEta) concatenate secondInstance.backpropagate(secondEta)
  }
  
  override def toString() = firstInstance.toString + " + " + secondInstance.toString
}

class ChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1, Type2] (first, second) {
  type Instance = InstanceOfChainNeuralNetwork[Type1,Type2]
  assert(first.inputDimension == second.outputDimension) 
  def inputDimension = second.inputDimension
  def outputDimension= first.outputDimension 
  def create(): InstanceOfChainNeuralNetwork[Type1, Type2] = new InstanceOfChainNeuralNetwork(this)
  override def toString() = "(" + first.toString + ") * (" + second.toString + ")" 
}

class InstanceOfChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(override val NN: ChainNeuralNetwork[Type1, Type2]) 
	extends InstanceOfMergedNeuralNetwork [Type1, Type2](NN) {
  type StructureType = ChainNeuralNetwork[Type1, Type2]

  def apply (x: NeuronVector) = {
    firstInstance(secondInstance(x)) 
  }
  
  def backpropagate(eta: NeuronVector) ={
    secondInstance.backpropagate(firstInstance.backpropagate(eta))
  }
  override def toString() = "(" + firstInstance.toString + ") * (" + secondInstance.toString + ")"
}

/********************************************************************************************/
// Basic neural network elements

/** SingleLayerNeuralNetwork is sigmoid functional layer 
 *  that takes in signals and transform them to activations [0,1] **/
class SingleLayerNeuralNetwork (override val dimension: Int, val func: NeuronFunction = SigmoidFunction /** Pointwise Function **/ ) 
	extends SelfTransform (dimension) {
  type InstanceType <: InstanceOfSingleLayerNeuralNetwork
  def create (): InstanceOfSingleLayerNeuralNetwork = new InstanceOfSingleLayerNeuralNetwork(this)
}
class InstanceOfSingleLayerNeuralNetwork (override val NN: SingleLayerNeuralNetwork) 
	extends InstanceOfSelfTransform (NN) with Memorable { 
  type StructureType <: SingleLayerNeuralNetwork
  
  def setWeights(seed:String, w:WeightVector) : Double = {0.0}
  def getWeights(seed:String) : NeuronVector = {NullVector}
  def getDerativeOfWeights(seed:String) : NeuronVector = {NullVector}
  
  private var gradient: NeuronVector = new NeuronVector(NN.dimension)
  
  var mirrorIndex :Int = 0
  def apply (x: NeuronVector) = {
    assert (x.length == inputDimension)
    inputBuffer(mirrorIndex) = x
    gradientBuffer(mirrorIndex) = NN.func.grad(x)
    outputBuffer(mirrorIndex) = NN.func(x)
        
    var cIndex = mirrorIndex
    mirrorIndex = (mirrorIndex + 1) % numOfMirrors
    outputBuffer(cIndex)
  }
  override def init(seed:String) = {
    if (status != seed) {
      status = seed
      numOfMirrors = 1 // find a new instance
      mirrorIndex = 0
    }
    else {      
      numOfMirrors = numOfMirrors + 1
    }
    this
  }
  
  var inputBuffer  = Array [NeuronVector]()
  var outputBuffer = Array [NeuronVector]()
  var gradientBuffer= Array [NeuronVector] ()
  
  override def allocate(seed:String) ={
    if (status == seed) {
      inputBuffer = new Array[NeuronVector] (numOfMirrors)
      outputBuffer= new Array[NeuronVector] (numOfMirrors)
      gradientBuffer= new Array[NeuronVector] (numOfMirrors)
      status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def backpropagate(eta: NeuronVector) = {
    val cIndex = mirrorIndex 
    mirrorIndex = (mirrorIndex + 1) % numOfMirrors
    eta DOT gradientBuffer(cIndex) // there is no penalty for sparsity
  }
}

/** SparseSingleLayer computes average activation and enforce sparsity penalty **/
class SparseSingleLayerNN (override val dimension: Int, 
						   var beta: Double = 0.0,
                           override val func: NeuronFunction = SigmoidFunction /** Pointwise Activation Function **/,
						   val penality: NeuronFunction = new KL_divergenceFunction(0.2) /** Sparsity Penalty Function **/)
	extends SingleLayerNeuralNetwork (dimension, func) {
  type InstanceType = InstanceOfSparseSingleLayerNN
  override def create (): InstanceOfSparseSingleLayerNN = new InstanceOfSparseSingleLayerNN(this)
} 

class InstanceOfSparseSingleLayerNN (override val NN: SparseSingleLayerNN) 
	extends InstanceOfSingleLayerNeuralNetwork (NN) {
  private var totalUsage: Int = 0 // reset if weights updated
  private var totalUsageOnUpdate: Int = 0
  override def setWeights(seed:String, w:WeightVector) : Double = {
    if (status != seed) {
      totalUsage = totalUsageOnUpdate
      totalUsageOnUpdate = 0
      rho := rhoOnUpdate
      rhoOnUpdate.set(0.0)
      if (totalUsage == 0) 0.0 /* use default value */ else NN.penality(rho / totalUsage).sum * NN.beta
    } else {
      0.0
    }
 }
  override def apply(x: NeuronVector) = {
    val y = super.apply(x)
    rhoOnUpdate += y; totalUsageOnUpdate = totalUsageOnUpdate + 1 // for computation of average activation
    y
  }
  private var rho : NeuronVector = new NeuronVector(outputDimension)
  private var rhoOnUpdate : NeuronVector = new NeuronVector(outputDimension)
  override def backpropagate(eta: NeuronVector) = {
    val cIndex = mirrorIndex 
    mirrorIndex = (mirrorIndex + 1) % numOfMirrors
    (eta + NN.penality.grad(rho/totalUsage) * NN.beta) DOT gradientBuffer(cIndex)
  }
  
  //def setBeta(b: Double): Null = {NN.beta = b; null}
}

/** LinearNeuralNetwork computes a linear transform, which is also possible to enforce L1/L2 regularization  **/
class LinearNeuralNetwork (inputDimension: Int, outputDimension: Int) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  type InstanceType <: InstanceOfLinearNeuralNetwork
  def create(): InstanceOfLinearNeuralNetwork = new InstanceOfLinearNeuralNetwork(this)
}
class InstanceOfLinearNeuralNetwork (override val NN: LinearNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) with Memorable {
  type StructureType <: LinearNeuralNetwork
  def setWeights(seed:String, wv:WeightVector) : Double = {
    if (status != seed) {
      status = seed
      wv(W, b) // get optimized weights
      dW.set(0.0) // reset derivative of weights
    }
    0.0
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      W.vec concatenate b 
    }else {
      NullVector
    }
  }
  def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      (dW.vec concatenate db) // / numOfMirrors
    } else {
      NullVector
    }
  }
  var mirrorIndex :Int = 0
  override def init(seed:String) = {
    if (status != seed) {
      status = seed
      numOfMirrors = 1
      mirrorIndex  = 0
    }
    else {      
      numOfMirrors = numOfMirrors + 1
      //println(numOfMirrors)
    }
    this
  }
  var inputBuffer  = Array [NeuronVector]()
  var outputBuffer = Array [NeuronVector]()
  override def allocate(seed:String) ={
    if (status == seed) {
      inputBuffer = new Array[NeuronVector] (numOfMirrors)
      outputBuffer= new Array[NeuronVector] (numOfMirrors)
      status = ""
    } else {}
    this
  }  
  protected val W: Weight = new Weight(outputDimension, inputDimension) 
  protected val b: NeuronVector = new NeuronVector (outputDimension)
  protected val dW:Weight = new Weight(outputDimension, inputDimension)
  protected val db:NeuronVector = new NeuronVector (outputDimension)
  def apply (x: NeuronVector) = {
    assert (x.length == inputDimension)
    inputBuffer(mirrorIndex) = x
    outputBuffer(mirrorIndex) = W* x + b
    var cIndex = mirrorIndex
    mirrorIndex = (mirrorIndex + 1) % numOfMirrors
    outputBuffer(cIndex)
  }

  def backpropagate(eta:NeuronVector) = {
    if (mirrorIndex == 0) { // start a new backpropagation
      dW.set(0.0); db.set(0.0)
    }
    dW+= eta CROSS inputBuffer(mirrorIndex)
    db+= eta
    mirrorIndex = (mirrorIndex + 1) % numOfMirrors
    W TransMult eta
  }
}

class RegularizedLinearNN (inputDimension: Int, outputDimension: Int, val lambda: Double)
	extends LinearNeuralNetwork (inputDimension, outputDimension) {
  type InstanceType = InstanceOfRegularizedLinearNN
  override def create(): InstanceOfRegularizedLinearNN = new InstanceOfRegularizedLinearNN(this) 
}

class InstanceOfRegularizedLinearNN (override val NN: RegularizedLinearNN) 
	extends InstanceOfLinearNeuralNetwork(NN) {
  type StructureType = RegularizedLinearNN
  override def setWeights(seed:String, wv:WeightVector) : Double = {
    if (status != seed) {
      status = seed
      wv(W, b) // get optimized weights
      dW.set(0.0) // reset derivative of weights
    }
    0.0 // + norm(W) * NN.lambda
  }
  override def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      (dW.vec concatenate db) // + (W / numOfMirrors, 0)
    } else {
      NullVector
    }
  }
}








