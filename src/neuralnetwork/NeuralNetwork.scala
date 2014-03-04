package neuralnetwork

abstract trait Vector {
  def concatenate (that : Vector) : Vector
  def divide(n: Int) : (Vector, Vector)
}

abstract trait Weight 

abstract trait Operationable  {
  def inputDimension:Int
  def outputDimension:Int
  
  implicit class JointHelper[T1<:Operationable](x:T1) {
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
  }
  implicit class ChainHelper[T1<:Operationable](x:T1) {
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
  }
  def create(): InstanceOfNeuralNetwork
}

abstract class NeuralNetwork (val inputDimension:Int, val outputDimension:Int) extends Operationable{
  def create() : InstanceOfNeuralNetwork
}
abstract class InstanceOfNeuralNetwork (val virtual: Operationable) extends Operationable {
  def inputDimension = virtual.inputDimension
  def outputDimension= virtual.outputDimension
  def create() = this // self reference
  def apply (x: Vector) : Vector
}




abstract class SelfTransform (val dimension: Int) extends NeuralNetwork(dimension, dimension) {
  def create() : InstanceOfSelfTransform
}
abstract class InstanceOfSelfTransform (virtual: SelfTransform) extends InstanceOfNeuralNetwork (virtual)


class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( val first:Type1, val second:Type2) 
	extends Operationable {
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension
  def create(): InstanceOfJointNeuralNetwork[Type1, Type2] = new InstanceOfJointNeuralNetwork(this)
}

class InstanceOfJointNeuralNetwork[Type1 <: Operationable, Type2 <:Operationable]
		(val NN: JointNeuralNetwork [Type1, Type2]) 
	extends InstanceOfNeuralNetwork (NN) {
  
  val firstInstance = NN.first.create()
  val secondInstance = NN.second.create()
  def apply (x: Vector)  = {
    var (first, second) = x.divide(NN.first.inputDimension)
    firstInstance(first).concatenate(
        secondInstance(second))
  }
}

class ChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		( val first:Type1, val second:Type2) 
	extends Operationable {
  def inputDimension = second.inputDimension
  def outputDimension= first.outputDimension
  def create(): InstanceOfChainNeuralNetwork[Type1, Type2] = new InstanceOfChainNeuralNetwork(this)
}

class InstanceOfChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(val NN: ChainNeuralNetwork[Type1, Type2]) 
	extends InstanceOfNeuralNetwork (NN) {
  val firstInstance  = NN.first.create()
  val secondInstance = NN.second.create()  
  def apply (x: Vector) = {
    firstInstance(secondInstance(x))
  }
}



class SingleLayerNeuralNetwork (val sigmod: Vector => Vector, override val dimension: Int)
	extends SelfTransform (dimension) {
  def create (): InstanceOfSingleLayerNeuralNetwork = new InstanceOfSingleLayerNeuralNetwork(this)
}
class InstanceOfSingleLayerNeuralNetwork (T: SingleLayerNeuralNetwork) 
	extends InstanceOfSelfTransform (T) { 
  def apply (x: Vector) = T.sigmod(x)
}


abstract trait Encoder {
  val encoder: InstanceOfNeuralNetwork
  def encode(x:Vector): Vector = encoder(x)
}
class AutoEncoder (val visible: SingleLayerNeuralNetwork, val hidden: SingleLayerNeuralNetwork) 
	extends SelfTransform (visible.dimension) {
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class InstanceOfAutoEncoder (val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with Encoder {
  val encoder = (NN.hidden TIMES NN.visible).create();
  val threeLayers = (NN.visible TIMES encoder).create();
  def apply (x:Vector) = threeLayers(x)
}

class RecursiveAutoEncoder(val word:SingleLayerNeuralNetwork) extends SelfTransform (word.dimension*2) {
  def create(): InstanceOfRecursiveAutoEncoder = new InstanceOfRecursiveAutoEncoder(this)
}
class InstanceOfRecursiveAutoEncoder(val NN: RecursiveAutoEncoder) 
	extends InstanceOfSelfTransform (NN) with Encoder {        
  val encoder = (NN.word TIMES (NN.word PLUS NN.word)).create()
  val threeLayers = ((NN.word PLUS NN.word) TIMES encoder).create()  
  def apply(x:Vector ) = threeLayers(x)
}

class ContextAwareRAE(val word:SingleLayerNeuralNetwork, val context: SingleLayerNeuralNetwork) 
extends NeuralNetwork (word.dimension*2+context.dimension, word.dimension*2) {
  def create = new InstanceOfContextAwareRAE(this);
}
class InstanceOfContextAwareRAE(val NN: ContextAwareRAE) 
	extends InstanceOfNeuralNetwork (NN) with Encoder{  
  val encoder = (NN.word  TIMES (NN.word PLUS NN.word PLUS NN.context)).create()
  val threeLayers = (((NN.word PLUS NN.word) TIMES encoder) PLUS NN.context).create()
  def apply(x:Vector) = threeLayers(x)
}








