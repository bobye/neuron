// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuralnetwork
import scala.collection.mutable._

/********************************************************************************************/
// Highest level classes for Neural Network
abstract trait Workspace{// 
  implicit class Helper[T1<:Operationable](x:T1) { 
    // Two basic operations to support combination 
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
    def SHARE [T2<:Operationable](y:T2) = new ShareNeuralNetwork(x,y)
    def REPEAT (n:Int) = new RepeatNeuralNetwork(x, n)
    def TENSOR [T2<:Operationable](y:T2) = // second order tensor only
      new TensorNeuralNetwork(x.outputDimension, y.outputDimension) TIMES (x PLUS y)
    def mTENSOR [T2<:Operationable](y:T2) = // mix first order and second order tensor
      (x TENSOR y) SHARE (x PLUS y)
  } 
}
/** Operationable is a generic trait that supports operations **/
abstract trait Operationable extends Workspace {
  def inputDimension:Int
  def outputDimension:Int
  
  val key:String = this.hashCode().toString

  def create(): InstanceOfNeuralNetwork
  def toStringGeneric(): String = key +
  	 "[" + inputDimension + "," + outputDimension + "]";
}

/** Memorable NN is instance that keep internal buffers **/
class SetOfMemorables extends HashMap[String, Memorable]
class Memorable {
  var status = ""
  var numOfMirrors:Int = 0
  var mirrorIndex: Int = 0
  //type arrayOfData[T<:NeuronVector] = Array[T]
  var inputBuffer  = Array [NeuronVector]()
  var outputBuffer = Array [NeuronVector]()
  var gradientBuffer= Array [NeuronVector] ()
  
  var inputBufferM = Array[NeuronMatrix]()
  var outputBufferM= Array[NeuronMatrix]()
  var gradientBufferM=Array[NeuronMatrix]()
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
  def apply (x: NeuronVector, mem:SetOfMemorables) : NeuronVector
  def apply (xs: NeuronMatrix, mem:SetOfMemorables) : NeuronMatrix 

  
  // structure to vectorization functions
  var status:String = ""

  def setWeights(seed:String, w:WeightVector) : Unit = {}// return regularization term
  def getWeights(seed:String): NeuronVector = NullVector // return internal weight as a vector
  def getRandomWeights(seed:String) : NeuronVector = NullVector // reset internal weights and return a vector
  def getDimensionOfWeights(seed: String): Int = 0
  def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = 0.0
  
  // dynamic operations
  def init(seed:String, mem: SetOfMemorables) : InstanceOfNeuralNetwork = {this} // default: do nothing
  def allocate(seed:String, mem: SetOfMemorables) : InstanceOfNeuralNetwork = {this} // 
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables): NeuronVector
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables): NeuronMatrix 
  
  // display
  override def toString() = "#" + toStringGeneric
}

abstract class SelfTransform (val dimension: Int) extends NeuralNetwork(dimension, dimension) 
abstract class InstanceOfSelfTransform (override val NN: SelfTransform) extends InstanceOfNeuralNetwork (NN)

class IdentityTransform(dimension: Int) extends SelfTransform(dimension) {
  type InstanceType = InstanceOfIdentityTransform
  def create(): InstanceOfIdentityTransform = new InstanceOfIdentityTransform(this)
  override def toString() = "" // print nothing
}
class InstanceOfIdentityTransform(override val NN:IdentityTransform) extends InstanceOfSelfTransform(NN) {
  def apply(x:NeuronVector, mem:SetOfMemorables) = x
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = xs
  def backpropagate(eta: NeuronVector, mem:SetOfMemorables) = eta
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = etas
}


// basic operation to derive hierarchy structures
abstract class MergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable] 
		(val first:Type1, val second:Type2) extends Operationable{ 
}

abstract class InstanceOfMergedNeuralNetwork [Type1 <:Operationable, Type2 <:Operationable]
		(override val NN: MergedNeuralNetwork[Type1, Type2]) 
		extends InstanceOfNeuralNetwork(NN) {
  val firstInstance = NN.first.create()
  val secondInstance = NN.second.create()
  
  
  override def setWeights(seed:String, w: WeightVector) : Unit = {
    if (status != seed) {
      status = seed
      firstInstance.setWeights(seed, w) 
      secondInstance.setWeights(seed, w)
    } else {
    }
  }
  override def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
        status = seed
        firstInstance.getWeights(seed) concatenate secondInstance.getWeights(seed)
      }else {
      NullVector
    }
  }
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
        status = seed
        firstInstance.getRandomWeights(seed) concatenate secondInstance.getRandomWeights(seed)
      }else {
      NullVector
    }
  }
  override def getDimensionOfWeights(seed: String): Int = {
    if (status != seed) {
      status = seed
      firstInstance.getDimensionOfWeights(seed) + secondInstance.getDimensionOfWeights(seed)
    } else {
      0
    }
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      firstInstance.getDerativeOfWeights(seed, dw, numOfSamples) +
      secondInstance.getDerativeOfWeights(seed, dw, numOfSamples)
    } else {
      0.0
    }
  }
  override def init(seed:String, mem:SetOfMemorables) = {
    firstInstance.init(seed, mem)
    secondInstance.init(seed, mem)
    this // do nothing
  }
  override def allocate(seed:String, mem:SetOfMemorables) = {
    firstInstance.allocate(seed, mem)
    secondInstance.allocate(seed, mem)
    this
  }
  
}
class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1,Type2](first,second) {
  type InstanceType <: InstanceOfJointNeuralNetwork[Type1,Type2]
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension 
  def create(): InstanceOfJointNeuralNetwork[Type1, Type2] = new InstanceOfJointNeuralNetwork(this)
  override def toString() = "(" + first.toString + " + " + second.toString + ")"
}

class RepeatNeuralNetwork [Type <:Operationable] (val x:Type, val n:Int) extends Operationable {
  assert (n>=1)
  val inputDimension = x.inputDimension * n
  val outputDimension = x.outputDimension * n
  def create() = n match {
    case 1 => x.create()
    case _ => (x PLUS new RepeatNeuralNetwork(x, n-1)).create()
  }
  override def toString() = "(" + x.toString() + "," + " n)"
}

class InstanceOfJointNeuralNetwork[Type1 <: Operationable, Type2 <:Operationable]
		(override val NN: JointNeuralNetwork [Type1, Type2]) 
	extends InstanceOfMergedNeuralNetwork [Type1, Type2](NN) {
  
  type StructureType <: JointNeuralNetwork[Type1, Type2]
  
  def apply (x: NeuronVector, mem:SetOfMemorables)  = {
    val (first, second) = x.splice(NN.first.inputDimension)
    // first compute secondInstance, then compute firstInstance
    // which is the inverse order of backpropagation
    val secondVec=  secondInstance(second, mem)
    val firstVec =  firstInstance(first, mem)
    firstVec concatenate secondVec
  }
  
  def apply (xs: NeuronMatrix, mem:SetOfMemorables) = {
    val (first, second) = xs.spliceRow(NN.first.inputDimension)
    val secondMat=  secondInstance(second, mem)
    val firstMat =  firstInstance(first, mem)
    firstMat padRow secondMat
  }

  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    val (firstEta, secondEta) = eta.splice(NN.first.outputDimension)
    firstInstance.backpropagate(firstEta, mem) concatenate secondInstance.backpropagate(secondEta, mem)
  }
  
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val (firstEta, secondEta) = etas.spliceRow(NN.first.outputDimension)
    firstInstance.backpropagate(firstEta, mem) padRow secondInstance.backpropagate(secondEta, mem)
  }
  
  override def toString() = firstInstance.toString + " + " + secondInstance.toString
}

class ChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1, Type2] (first, second) {
  type InstanceType <: InstanceOfChainNeuralNetwork[Type1,Type2]
  assert(first.inputDimension == second.outputDimension) 
  def inputDimension = second.inputDimension
  def outputDimension= first.outputDimension 
  def create(): InstanceOfChainNeuralNetwork[Type1, Type2] = new InstanceOfChainNeuralNetwork(this)
  override def toString() = "(" + first.toString + ") * (" + second.toString + ")" 
}

class InstanceOfChainNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable] 
		(override val NN: ChainNeuralNetwork[Type1, Type2]) 
	extends InstanceOfMergedNeuralNetwork [Type1, Type2](NN) {
  type StructureType <: ChainNeuralNetwork[Type1, Type2]

  def apply (x: NeuronVector, mem:SetOfMemorables) = {
    // first compute secondInstance, then compute firstInstance
    // which is the inverse order of backpropagation
    firstInstance(secondInstance(x, mem), mem) 
  }
  def apply(xs: NeuronMatrix, mem:SetOfMemorables) = {
    firstInstance(secondInstance(xs, mem), mem)
  }
  
  def backpropagate(eta: NeuronVector, mem:SetOfMemorables) ={
    secondInstance.backpropagate(firstInstance.backpropagate(eta, mem), mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    secondInstance.backpropagate(firstInstance.backpropagate(etas, mem), mem)
  }
  override def toString() = "(" + firstInstance.toString + ") * (" + secondInstance.toString + ")"
}

class ShareNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val first:Type1, override val second: Type2)
		extends MergedNeuralNetwork[Type1, Type2] (first, second) {
  assert (first.inputDimension == second.inputDimension)
  type InstanceType <: InstanceOfShareNeuralNetwork[Type1, Type2]
  def inputDimension = first.inputDimension
  def outputDimension = first.outputDimension + second.outputDimension
  def create():InstanceOfShareNeuralNetwork[Type1,Type2] = new InstanceOfShareNeuralNetwork(this)
  override def toString() = "(" + first.toString + ") _ (" + second.toString() + ")"
}

class InstanceOfShareNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val NN: ShareNeuralNetwork[Type1, Type2])
		extends InstanceOfMergedNeuralNetwork(NN) {
  type StructureType <: ShareNeuralNetwork[Type1, Type2]
  def apply(x:NeuronVector, mem: SetOfMemorables) = {
    val secondVec = secondInstance(x, mem)
    val firstVec  = firstInstance(x, mem)
    firstVec concatenate secondVec
  }
  def apply(x:NeuronMatrix, mem:SetOfMemorables) = {
    val secondMat = secondInstance(x, mem)
    val firstMat = firstInstance(x, mem)
    firstMat padRow secondMat
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    val (firstEta, secondEta) = eta.splice(NN.first.outputDimension)
    firstInstance.backpropagate(firstEta, mem) + secondInstance.backpropagate(secondEta, mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
	val (firstEta, secondEta) = etas.spliceRow(NN.first.outputDimension)
	firstInstance.backpropagate(firstEta, mem) + secondInstance.backpropagate(secondEta, mem)
  }
  override def toString() = "(" + firstInstance.toString + ") _ (" + secondInstance.toString + ")"
}






