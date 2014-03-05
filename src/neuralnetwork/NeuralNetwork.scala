package neuralnetwork

/********************************************************************************************/
// Numerical classes and their operation 
abstract trait Scalar extends Vector
abstract trait Vector {
  def concatenate (that : Vector) : Vector
  def divide(n: Int) : (Vector, Vector)
  def -(that:Vector): Vector
  def +(that:Vector): Vector
  def DOT(that: Vector): Vector
}
trait Weight {
  def *(x:Vector):Vector
  def Mult(x:Vector) = this * x
  def TransMult(x:Vector): Vector
}

object NullVector extends Vector {
  def concatenate(that: Vector) = that
  def divide(n: Int) = (NullVector, NullVector)
  def -(that:Vector) = NullVector
  def +(that:Vector) = NullVector
  def DOT(that:Vector) = NullVector
}

object NullWeight extends Weight {
  def *(x:Vector) = NullVector
  def TransMult(x:Vector) = NullVector
}

abstract trait Function {
  def grad(x:Vector): Vector
  def apply(x:Vector): Vector
}

object IndentityFunction extends Function {
  def grad(x:Vector): Vector = NullVector
  def apply(x:Vector) = x
}


/********************************************************************************************/
// Highest level classes for Neural Network
abstract trait Workspace {
  implicit class Helper[T1<:Operationable](x:T1) {
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
  } 
}
abstract trait Operationable extends Workspace {
  def inputDimension:Int
  def outputDimension:Int

  def create(): InstanceOfNeuralNetwork
  override def toString(): String = this.hashCode().toString
}

// Class for neural network template
abstract class NeuralNetwork (val inputDimension:Int, val outputDimension:Int) extends Operationable{
  def create() : InstanceOfNeuralNetwork
  override def toString() = "?" + this.hashCode().toString
}

// Class for neural network instance, which can be applied and trained
abstract class InstanceOfNeuralNetwork (val NN: Operationable) extends Operationable {
  def inputDimension = NN.inputDimension
  def outputDimension= NN.outputDimension
  def create() = this // self reference
  def apply (x: Vector) : Vector
  
  def init(rand: => Weight) : InstanceOfNeuralNetwork
  def backpropagate(eta: Vector): Vector
  def train (x: Vector, y: Vector) : InstanceOfNeuralNetwork = {
    backpropagate(apply(x) - y)
    this
  }
  override def toString() = "#" + this.hashCode().toString
}

abstract class SelfTransform (val dimension: Int) extends NeuralNetwork(dimension, dimension) 
abstract class InstanceOfSelfTransform (override val NN: SelfTransform) extends InstanceOfNeuralNetwork (NN)




// basic operation to derive hierarchy structures
class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( val first:Type1, val second:Type2) 
	extends Operationable {
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension
  def create(): InstanceOfJointNeuralNetwork[Type1, Type2] = new InstanceOfJointNeuralNetwork(this)
  override def toString() = "(" + first.toString + " + " + second.toString + ")"
}

class InstanceOfJointNeuralNetwork[Type1 <: Operationable, Type2 <:Operationable]
		(override val NN: JointNeuralNetwork [Type1, Type2]) 
	extends InstanceOfNeuralNetwork (NN) {
  
  val firstInstance = NN.first.create()
  val secondInstance = NN.second.create()
  def apply (x: Vector)  = {
    var (first, second) = x.divide(NN.first.inputDimension)
    firstInstance(first) concatenate secondInstance(second)
  }
  def init(rand: => Weight) = {
    firstInstance.init(rand)
    secondInstance.init(rand)
    this // do nothing
  }
  def backpropagate(eta: Vector) = {
    var (firstEta, secondEta) = eta.divide(NN.first.inputDimension)
    firstInstance.backpropagate(firstEta) concatenate secondInstance.backpropagate(secondEta)
  }
  
  override def toString() = firstInstance.toString + " + " + secondInstance.toString
}

class ChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		( val first:Type1, val second:Type2) 
	extends Operationable {
  def inputDimension = second.inputDimension
  def outputDimension= first.outputDimension
  def create(): InstanceOfChainNeuralNetwork[Type1, Type2] = new InstanceOfChainNeuralNetwork(this)
  override def toString() = first.toString + " * (" + second.toString + ")" 
}

class InstanceOfChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(override val NN: ChainNeuralNetwork[Type1, Type2]) 
	extends InstanceOfNeuralNetwork (NN) {
  val firstInstance  = NN.first.create()
  val secondInstance = NN.second.create()  
  def apply (x: Vector) = {
    firstInstance(secondInstance(x)) // need to multiply W
  }
  def init(rand: => Weight) = {
    firstInstance.init(rand)
    secondInstance.init(rand)
    this
  }
  
  def backpropagate(eta: Vector) ={
    secondInstance.backpropagate(firstInstance.backpropagate(eta))
  }
  override def toString() = firstInstance.toString + " * (" + secondInstance.toString + ")"
}

/********************************************************************************************/
// basic neural network elements
class SingleLayerNeuralNetwork (val func: Function, override val dimension: Int)
	extends SelfTransform (dimension) {
  def create (): InstanceOfSingleLayerNeuralNetwork = new InstanceOfSingleLayerNeuralNetwork(this)
}
class InstanceOfSingleLayerNeuralNetwork (override val NN: SingleLayerNeuralNetwork) 
	extends InstanceOfSelfTransform (NN) { 
  private var gradient: Vector = NullVector
  def apply (x: Vector) = {
    gradient = NN.func.grad(x)
    NN.func(x)
  }
  def init(rand: => Weight) = {this}
  def backpropagate(eta: Vector) = eta DOT gradient
}

class LinearNeuralNetwork (inputDimension: Int, outputDimension: Int) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  def create(): InstanceOfLinearNeuralNetwork = new InstanceOfLinearNeuralNetwork(this)
}
class InstanceOfLinearNeuralNetwork (override val NN: LinearNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) {
  private var W: Weight = NullWeight
  def apply (x: Vector) = W*x
  def init(rand: => Weight) = {
    W = rand
    this
  }
  def backpropagate(eta:Vector) = W TransMult eta
}

/********************************************************************************************/
// Encoder and Recursive Neural Network
abstract trait EncodeClass {
  val encodeDimension: Int
}
abstract trait Encoder extends Operationable with EncodeClass{
  override def create(): InstanceOfEncoder
} 
abstract trait InstanceOfEncoder extends InstanceOfNeuralNetwork with EncodeClass{
  val encoder: InstanceOfNeuralNetwork
  def encode(x:Vector): Vector = encoder(x)
}

// It implicitly requires the dimensional constraints
abstract trait RecursiveEncoder extends Encoder {
  override def create(): InstanceOfRecursiveEncoder
}
abstract trait InstanceOfRecursiveEncoder extends InstanceOfEncoder

// Convert Encoder and InstanceOfEncoder to new NeuralNetwork by replacing apply() with encode()
class EncoderNeuralNetwork [T<: Encoder] (val NN: T) extends NeuralNetwork(NN.inputDimension, NN.encodeDimension) {
  def create() = new InstanceOfEncoderNeuralNetwork(this, NN.create())
}

class InstanceOfEncoderNeuralNetwork [T1<: Encoder, T2<: InstanceOfEncoder] // T1 and T2 must be compatible
		(override val NN: EncoderNeuralNetwork[T1], val INN: T2) 
	extends InstanceOfNeuralNetwork (INN.NN) {
  override val outputDimension = INN.encodeDimension
  def apply (x:Vector) = INN.encode(x)
  def init(rand: =>Weight) = INN.init(rand)
  def backpropagate(eta:Vector) = INN.encoder.backpropagate(eta)
}

// AutoEncoder
class AutoEncoder (override val dimension: Int, val hidden: NeuralNetwork)
	extends SelfTransform (dimension) with Encoder {
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class InstanceOfAutoEncoder (override val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with InstanceOfEncoder {
  private val inputLayer = new LinearNeuralNetwork(NN.dimension, NN.hidden.inputDimension)
  private val outputLayer = new LinearNeuralNetwork(NN.hidden.outputDimension, NN.dimension)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val threeLayers = (outputLayer TIMES encoder).create()
  def apply (x:Vector) = threeLayers(x)
  def init(rand: => Weight) = {
    threeLayers.init(rand)
    this
  }
  def backpropagate(eta:Vector) = threeLayers.backpropagate(eta)
}

class SingleLayerAutoEncoder (val func:Function) (override val dimension:Int, val hiddenDimension:Int) 
	extends AutoEncoder(dimension, new SingleLayerNeuralNetwork(func, hiddenDimension))

class RecursiveSingleLayerAE (override val func:Function) (val wordLength: Int) 
	extends SingleLayerAutoEncoder(func)(wordLength*2, wordLength) with RecursiveEncoder {
  override def create() : InstanceOfRecursiveSingleLayerAE = new InstanceOfRecursiveSingleLayerAE(this)
}
class InstanceOfRecursiveSingleLayerAE(override val NN:RecursiveSingleLayerAE) 
	extends InstanceOfAutoEncoder(NN) with InstanceOfRecursiveEncoder

//Context Aware Auto Encoder
class ContextAwareAutoEncoder(val codeLength: Int, val contextLength: Int, val hidden: NeuralNetwork) 
	extends SelfTransform(codeLength + contextLength) with Encoder {
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfContextAwareAutoEncoder = new InstanceOfContextAwareAutoEncoder(this)
}
class InstanceOfContextAwareAutoEncoder(override val NN:ContextAwareAutoEncoder) 
	extends InstanceOfSelfTransform(NN) with InstanceOfEncoder {
  private val inputLayer = new LinearNeuralNetwork(NN.codeLength +NN.contextLength, NN.hidden.inputDimension)
  private val finalLayer = new LinearNeuralNetwork(NN.hidden.outputDimension + NN.contextLength, NN.codeLength)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val topLayer = finalLayer.create()
  def apply (x:Vector) = {
    var (_, context) = x.divide(NN.codeLength)
    topLayer(encoder(x) concatenate context) concatenate context
  } 
  def init(rand: => Weight) = {
    encoder.init(rand)
    topLayer.init(rand)
    this
  }
  def backpropagate(eta:Vector) = {
    var (eta_31, _) = eta.divide(NN.codeLength)
    var (eta_21, _) = topLayer.backpropagate(eta_31).divide(NN.hidden.outputDimension)
    encoder.backpropagate(eta_21)
  }
  
}

class SingleLayerCAE(val func: Function) (override val codeLength: Int, override val contextLength:Int, val hiddenDimension: Int)
	extends ContextAwareAutoEncoder(codeLength, contextLength, new SingleLayerNeuralNetwork(func, hiddenDimension))

class RecursiveSingleLayerCAE (override val func: Function) (val wordLength:Int, override val contextLength: Int)
	extends SingleLayerCAE(func)(wordLength*2, contextLength, wordLength) with RecursiveEncoder {
  override def create(): InstanceOfRecursiveSingleLayerCAE = new InstanceOfRecursiveSingleLayerCAE(this)
}
class InstanceOfRecursiveSingleLayerCAE (override val NN: RecursiveSingleLayerCAE)
	extends InstanceOfContextAwareAutoEncoder(NN) with InstanceOfRecursiveEncoder








