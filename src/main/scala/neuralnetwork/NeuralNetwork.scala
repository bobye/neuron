// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updates: Mar 2014
// Copyright under MIT License
package neuralnetwork

/********************************************************************************************/
// Numerical classes and their operations 
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix


class NeuronVector (val data: DenseVector[Float]){
  def this(n:Int) = this(DenseVector.zeros[Float] (n))
  def concatenate (that: NeuronVector) : NeuronVector = new NeuronVector(DenseVector.vertcat(this.data, that.data))
  def splice(num: Int) : (NeuronVector, NeuronVector) = (new NeuronVector(this.data(0 to num-1)), new NeuronVector(this.data(num to -1)))

  def -(that:NeuronVector): NeuronVector = new NeuronVector(this.data - that.data)
  def +(that:NeuronVector): NeuronVector = new NeuronVector(this.data + that.data)
  def DOT(that: NeuronVector): NeuronVector = new NeuronVector(this.data :* that.data)
}
class Weight (val data:DenseMatrix[Float]){
  def this(rows:Int, cols:Int) = this (DenseMatrix.zeros[Float](rows, cols))
  def *(x:NeuronVector):NeuronVector = new NeuronVector(data * x.data)
  def Mult(x:NeuronVector) = this * x
  def TransMult(x:NeuronVector): NeuronVector = new NeuronVector(this.data.t * x.data)
}

object NullVector extends NeuronVector (0)

object NullWeight extends Weight (0,0)

abstract trait NeuronFunction {
  def grad(x:NeuronVector): NeuronVector
  def apply(x:NeuronVector): NeuronVector
}

object IndentityFunction extends NeuronFunction {
  def grad(x:NeuronVector): NeuronVector = new NeuronVector(DenseVector.ones[Float](x.data.length))
  def apply(x:NeuronVector) = x
}
/*
object SigmoidFunction extends NeuronFunction {
  def grad(x:NeuronVector): NeuronVector
  def apply(x:NeuronVector): NeuronVector
}
*/
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
  def apply (x: NeuronVector) : NeuronVector
  
  def init(rand: => Weight) : InstanceOfNeuralNetwork
  def backpropagate(eta: NeuronVector): NeuronVector
  def train (x: NeuronVector, y: NeuronVector) : InstanceOfNeuralNetwork = {
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
  def apply (x: NeuronVector)  = {
    var (first, second) = x.splice(NN.first.inputDimension)
    firstInstance(first) concatenate secondInstance(second)
  }
  def init(rand: => Weight) = {
    firstInstance.init(rand)
    secondInstance.init(rand)
    this // do nothing
  }
  def backpropagate(eta: NeuronVector) = {
    var (firstEta, secondEta) = eta.splice(NN.first.inputDimension)
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
  def apply (x: NeuronVector) = {
    firstInstance(secondInstance(x)) // need to multiply W
  }
  def init(rand: => Weight) = {
    firstInstance.init(rand)
    secondInstance.init(rand)
    this
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
  def create (): InstanceOfSingleLayerNeuralNetwork = new InstanceOfSingleLayerNeuralNetwork(this)
}
class InstanceOfSingleLayerNeuralNetwork (override val NN: SingleLayerNeuralNetwork) 
	extends InstanceOfSelfTransform (NN) { 
  private var gradient: NeuronVector = new NeuronVector(NN.dimension)
  def apply (x: NeuronVector) = {
    gradient = NN.func.grad(x)
    NN.func(x)
  }
  def init(rand: => Weight) = {this}
  def backpropagate(eta: NeuronVector) = eta DOT gradient
}

class LinearNeuralNetwork (inputDimension: Int, outputDimension: Int) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  def create(): InstanceOfLinearNeuralNetwork = new InstanceOfLinearNeuralNetwork(this)
}
class InstanceOfLinearNeuralNetwork (override val NN: LinearNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) {
  private var W: Weight = new Weight(NN.outputDimension, NN.inputDimension)
  def apply (x: NeuronVector) = W*x
  def init(rand: => Weight) = {
    W = rand
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
  override def create(): InstanceOfEncoder
} 
abstract trait InstanceOfEncoder extends InstanceOfNeuralNetwork with EncodeClass{
  val encoder: InstanceOfNeuralNetwork
  def encode(x:NeuronVector): NeuronVector = encoder(x)
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
  def apply (x:NeuronVector) = INN.encode(x)
  def init(rand: =>Weight) = INN.init(rand)
  def backpropagate(eta:NeuronVector) = INN.encoder.backpropagate(eta)
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
  def apply (x:NeuronVector) = threeLayers(x)
  def init(rand: => Weight) = {
    threeLayers.init(rand)
    this
  }
  def backpropagate(eta:NeuronVector) = threeLayers.backpropagate(eta)
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
  def apply (x:NeuronVector) = {
    var (_, context) = x.splice(NN.codeLength)
    topLayer(encoder(x) concatenate context) concatenate context
  } 
  def init(rand: => Weight) = {
    encoder.init(rand)
    topLayer.init(rand)
    this
  }
  def backpropagate(eta:NeuronVector) = {
    var (eta_31, _) = eta.splice(NN.codeLength)
    var (eta_21, _) = topLayer.backpropagate(eta_31).splice(NN.hidden.outputDimension)
    encoder.backpropagate(eta_21)
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








