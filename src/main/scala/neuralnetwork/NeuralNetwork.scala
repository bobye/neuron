// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updates: Mar 2014
// Copyright under MIT License
package neuralnetwork

/********************************************************************************************/
// Numerical classes and their operations 
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
//import breeze.math._

class NeuronVector (val data: DenseVector[Double]) {
  def this(n:Int) = this(DenseVector.zeros[Double] (n))
  def concatenate (that: NeuronVector) : NeuronVector = new NeuronVector(DenseVector.vertcat(this.data, that.data))
  def splice(num: Int) : (NeuronVector, NeuronVector) = (new NeuronVector(this.data(0 until num)), new NeuronVector(this.data(num to -1)))

  def -(that:NeuronVector): NeuronVector = new NeuronVector(this.data - that.data)
  def +(that:NeuronVector): NeuronVector = new NeuronVector(this.data + that.data)
  def DOT(that: NeuronVector): NeuronVector = new NeuronVector(this.data :* that.data)
  
  def set(x:Double) : Null = {data:=x; null}
}
class Weight (val data:DenseMatrix[Double]){
  def this(rows:Int, cols:Int) = this(DenseMatrix.zeros[Double](rows,cols))
  def *(x:NeuronVector):NeuronVector = new NeuronVector(data * x.data)
  def Mult(x:NeuronVector) = this * x
  def TransMult(x:NeuronVector): NeuronVector = new NeuronVector(this.data.t * x.data)
  def vec = new NeuronVector(data.toDenseVector)
}

class WeightVector (override val data: DenseVector[Double]) extends NeuronVector(data) {
  var ptr : Int = 0
  def reset(): Null = {ptr = 0; null}
  def apply(W:Weight): Int = {
    var rows = W.data.rows
    var cols = W.data.cols
    
    W.data := (data(ptr until ptr + rows*cols).asDenseMatrix.reshape(rows, cols))
    //new Weight((data(ptr-rows*cols until ptr).asDenseMatrix.reshape(rows, cols)))
    ptr = ptr + rows * cols
    ptr
  }
  def set(wv: NeuronVector): Int = {
    ptr = 0
    data := wv.data
    0
  }
}


object NullVector extends NeuronVector (0)

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
  
object SigmoidFunction extends NeuronFunction {
  def grad(x:NeuronVector): NeuronVector = new NeuronVector(dsgm(x.data))
  def apply(x:NeuronVector): NeuronVector= new NeuronVector(sigmoid(x.data))
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
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
  } 
}
abstract trait Operationable extends Workspace {
  def inputDimension:Int
  def outputDimension:Int

  def create(): InstanceOfNeuralNetwork
  def toStringGeneric(): String = this.hashCode().toString +
  	 "[" + inputDimension + "," + outputDimension + "]";
}

// Class for neural network template
abstract class NeuralNetwork (val inputDimension:Int, val outputDimension:Int) extends Operationable{
  type InstanceType <: InstanceOfNeuralNetwork
  def create() : InstanceOfNeuralNetwork 
  override def toString() = "?" + toStringGeneric
}

// Class for neural network instance, which can be applied and trained
abstract class InstanceOfNeuralNetwork (val NN: Operationable) extends Operationable {
  type StructureType <: Operationable
  // basic topological structure
  def inputDimension = NN.inputDimension
  def outputDimension= NN.outputDimension
  def create() = this // self reference
  def apply (x: NeuronVector) : NeuronVector
  
  // structure to vectorization functions
  var status:String = ""

  def setWeights(seed:String, w:WeightVector) : InstanceOfNeuralNetwork
  def getWeights(seed:String) : NeuronVector
  def getDerativeOfWeights(seed:String) : NeuronVector
  
  // dynamic operations
  def init() : InstanceOfNeuralNetwork
  def backpropagate(eta: NeuronVector): NeuronVector
  
  
  def train (x: NeuronVector, y: NeuronVector) : InstanceOfNeuralNetwork = {
    // it has many things to do
    backpropagate(apply(x) - y)
    this
  }
  
  // display
  override def toString() = "#" + toStringGeneric
}

abstract class SelfTransform (val dimension: Int) extends NeuralNetwork(dimension, dimension) 
abstract class InstanceOfSelfTransform (override val NN: SelfTransform) extends InstanceOfNeuralNetwork (NN)




// basic operation to derive hierarchy structures
abstract class MergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable] 
		(val first:Type1, val second:Type2) extends Operationable{
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension  
}

abstract class InstanceOfMergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable]
		(override val NN: MergedNeuralNetwork[Type1, Type2]) 
		extends InstanceOfNeuralNetwork(NN) {
  val firstInstance = NN.first.create()
  val secondInstance = NN.second.create()
  
  
  override def setWeights(seed:String, w: WeightVector) : InstanceOfMergedNeuralNetwork[Type1, Type2] = {
    if (status != seed) {
      firstInstance.setWeights(seed, w)
      secondInstance.setWeights(seed, w)
    } else {
      status = seed
    }
    this
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      firstInstance.getWeights(seed) concatenate secondInstance.getWeights(seed)
    }else {
      NullVector
    }
  }
  def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      firstInstance.getDerativeOfWeights(seed) concatenate secondInstance.getDerativeOfWeights(seed)
    } else {
      NullVector
    }
  }
  def init() = {
    firstInstance.init()
    secondInstance.init()
    this // do nothing
  }
  
}
class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1,Type2](first,second) {
  type Instance = InstanceOfJointNeuralNetwork[Type1,Type2]
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
  def create(): InstanceOfChainNeuralNetwork[Type1, Type2] = new InstanceOfChainNeuralNetwork(this)
  override def toString() = first.toString + " * (" + second.toString + ")" 
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
  override def toString() = firstInstance.toString + " * (" + secondInstance.toString + ")"
}

/********************************************************************************************/
// Basic neural network elements (Only two for now)
class SingleLayerNeuralNetwork (val func: NeuronFunction /** Pointwise Function **/, override val dimension: Int) 
	extends SelfTransform (dimension) {
  type InstanceType = InstanceOfSingleLayerNeuralNetwork
  def create (): InstanceOfSingleLayerNeuralNetwork = new InstanceOfSingleLayerNeuralNetwork(this)
}
class InstanceOfSingleLayerNeuralNetwork (override val NN: SingleLayerNeuralNetwork) 
	extends InstanceOfSelfTransform (NN) { 
  type StructureType = SingleLayerNeuralNetwork
  
  def setWeights(seed:String, w:WeightVector) : InstanceOfSingleLayerNeuralNetwork = {this}
  def getWeights(seed:String) : NeuronVector = {NullVector}
  def getDerativeOfWeights(seed:String) : NeuronVector = {NullVector}
  
  private var gradient: NeuronVector = new NeuronVector(NN.dimension)
  def apply (x: NeuronVector) = {
    gradient = NN.func.grad(x)
    NN.func(x)
  }
  def init() = {this}
  def backpropagate(eta: NeuronVector) = eta DOT gradient
}

class LinearNeuralNetwork (inputDimension: Int, outputDimension: Int) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  type InstanceType = InstanceOfLinearNeuralNetwork
  def create(): InstanceOfLinearNeuralNetwork = new InstanceOfLinearNeuralNetwork(this)
}
class InstanceOfLinearNeuralNetwork (override val NN: LinearNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) {
  type StructureType = LinearNeuralNetwork
  def setWeights(seed:String, wv:WeightVector) : InstanceOfLinearNeuralNetwork = {
    if (status != seed) {
      status = seed
      wv(W)
    }
    this
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      W.vec
    }else {
      NullVector
    }
  }
  def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      dW.vec
    } else {
      NullVector
    }
  }
  
  private val W: Weight = new Weight(outputDimension, inputDimension)
  private val dW:Weight = new Weight(outputDimension, inputDimension)
  def apply (x: NeuronVector) = W*x
  def init() = {
    this
  }
  def backpropagate(eta:NeuronVector) = W TransMult eta
}

/********************************************************************************************/
// Encoder and Recursive Neural Network
abstract trait EncodeClass {
  val encodeDimension: Int
}
abstract trait Encoder extends Operationable with EncodeClass{
  type InstanceType <: InstanceOfEncoder
  override def create(): InstanceOfEncoder
} 
abstract trait InstanceOfEncoder extends InstanceOfNeuralNetwork with EncodeClass{
  type StructureType <: Encoder
  val encoder: InstanceOfNeuralNetwork
  def encode(x:NeuronVector): NeuronVector = encoder(x)
}

// It implicitly requires the dimensional constraints
abstract trait RecursiveEncoder extends Encoder {
  override def create(): InstanceOfRecursiveEncoder
}
abstract trait InstanceOfRecursiveEncoder extends InstanceOfEncoder

// Convert Encoder and InstanceOfEncoder to new NeuralNetwork by replacing apply() with encode()
// However, the weights of both are not changed, which means in back propagation 
// only part of weights are got updated
class EncoderNeuralNetwork [T<: Encoder] (val NN: T) extends NeuralNetwork(NN.inputDimension, NN.encodeDimension) {
  type InstanceType <: InstanceOfEncoderNeuralNetwork [NN.InstanceType]
  def create() = new InstanceOfEncoderNeuralNetwork(NN.create())
}

class InstanceOfEncoderNeuralNetwork [T<: InstanceOfEncoder] // T1 and T2 must be compatible
		(val INN: T) 
	extends InstanceOfNeuralNetwork (INN.NN) {
  type StructureType <: EncoderNeuralNetwork[INN.StructureType]  
  
  override val outputDimension = INN.encodeDimension
  def apply (x:NeuronVector) = INN.encode(x)
  def init() = INN.init()
  def backpropagate(eta:NeuronVector) = INN.encoder.backpropagate(eta)
  
  def setWeights(seed:String, w:WeightVector) = INN.setWeights(seed, w)
  def getWeights(seed:String) : NeuronVector = INN.getWeights(seed)
  def getDerativeOfWeights(seed:String) : NeuronVector = INN.getDerativeOfWeights(seed)
}

// AutoEncoder
class AutoEncoder (override val dimension: Int, val hidden: NeuralNetwork)
	extends SelfTransform (dimension) with Encoder {
  type InstanceType <: InstanceOfAutoEncoder
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class InstanceOfAutoEncoder (override val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with InstanceOfEncoder {
  type Structure <: AutoEncoder
  private val inputLayer = new LinearNeuralNetwork(NN.dimension, NN.hidden.inputDimension)
  private val outputLayer = new LinearNeuralNetwork(NN.hidden.outputDimension, NN.dimension)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val threeLayers = (outputLayer TIMES encoder).create()
  def apply (x:NeuronVector) = threeLayers(x)
  def init() = {
    threeLayers.init()
    this
  }
  def backpropagate(eta:NeuronVector) = threeLayers.backpropagate(eta)
  
  def setWeights(seed:String, w:WeightVector): InstanceOfAutoEncoder = { threeLayers.setWeights(seed, w); this}
  def getWeights(seed:String) : NeuronVector = threeLayers.getWeights(seed)
  def getDerativeOfWeights(seed:String) : NeuronVector = threeLayers.getDerativeOfWeights(seed)
}

class SingleLayerAutoEncoder (val func:NeuronFunction) (override val dimension:Int, val hiddenDimension:Int) 
	extends AutoEncoder(dimension, new SingleLayerNeuralNetwork(func, hiddenDimension))

class RecursiveSingleLayerAE (override val func:NeuronFunction) (val wordLength: Int) 
	extends SingleLayerAutoEncoder(func)(wordLength*2, wordLength) with RecursiveEncoder {
  override def create() : InstanceOfRecursiveSingleLayerAE = new InstanceOfRecursiveSingleLayerAE(this)
}
class InstanceOfRecursiveSingleLayerAE(override val NN:RecursiveSingleLayerAE) 
	extends InstanceOfAutoEncoder(NN) with InstanceOfRecursiveEncoder

//Context Aware Auto Encoder
class ContextAwareAutoEncoder(val codeLength: Int, val contextLength: Int, val hidden: NeuralNetwork) 
	extends SelfTransform(codeLength + contextLength) with Encoder {
  type Instance <: InstanceOfContextAwareAutoEncoder
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfContextAwareAutoEncoder = new InstanceOfContextAwareAutoEncoder(this)
}
class InstanceOfContextAwareAutoEncoder(override val NN:ContextAwareAutoEncoder) 
	extends InstanceOfSelfTransform(NN) with InstanceOfEncoder {
  type Structure <: ContextAwareAutoEncoder
  private val inputLayer = new LinearNeuralNetwork(NN.codeLength +NN.contextLength, NN.hidden.inputDimension)
  private val finalLayer = new LinearNeuralNetwork(NN.hidden.outputDimension + NN.contextLength, NN.codeLength)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val topLayer = finalLayer.create()
  def apply (x:NeuronVector) = {
    var (_, context) = x.splice(NN.codeLength)
    topLayer(encoder(x) concatenate context) concatenate context
  } 
  def init() = {
    encoder.init()
    topLayer.init()
    this
  }
  def backpropagate(eta:NeuronVector) = {
    var (eta_31, _) = eta.splice(NN.codeLength)
    var (eta_21, _) = topLayer.backpropagate(eta_31).splice(NN.hidden.outputDimension)
    encoder.backpropagate(eta_21)
  }
  
  def setWeights(seed:String, w:WeightVector): InstanceOfContextAwareAutoEncoder = {
    if (status != seed) {
    	topLayer.setWeights(seed, w)
    	encoder.setWeights(seed, w)
    }
    this
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      topLayer.getWeights(seed) concatenate encoder.getWeights(seed)
    } else NullVector
  }
  def getDerativeOfWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      topLayer.getDerativeOfWeights(seed) concatenate encoder.getDerativeOfWeights(seed)
    } else NullVector
  }
  
}

class SingleLayerCAE(val func: NeuronFunction) (override val codeLength: Int, override val contextLength:Int, val hiddenDimension: Int)
	extends ContextAwareAutoEncoder(codeLength, contextLength, new SingleLayerNeuralNetwork(func, hiddenDimension))

class RecursiveSingleLayerCAE (override val func: NeuronFunction) (val wordLength:Int, override val contextLength: Int)
	extends SingleLayerCAE(func)(wordLength*2, contextLength, wordLength) with RecursiveEncoder {
  override def create(): InstanceOfRecursiveSingleLayerCAE = new InstanceOfRecursiveSingleLayerCAE(this)
}
class InstanceOfRecursiveSingleLayerCAE (override val NN: RecursiveSingleLayerCAE)
	extends InstanceOfContextAwareAutoEncoder(NN) with InstanceOfRecursiveEncoder








