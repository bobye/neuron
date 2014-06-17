// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuron.core
import scala.collection.mutable._
import neuron.math._

/******************************************************************************************/
// Highest level classes for Neural Network

/** defined operations for Operationable neural network (or its instance) */
abstract trait Workspace{// 
  implicit class Helper[T1<:Operationable](x:T1) { 
    // Two basic operations to support combination 
    def PLUS [T2<:Operationable](y:T2) = new JointNeuralNetwork(x,y)
    def TIMES [T2<:Operationable](y:T2) = new ChainNeuralNetwork(x,y)
    def SHARE [T2<:Operationable](y:T2) = new ShareNeuralNetwork(x,y)
    def ADD [T2<:Operationable](y:T2) = new AddedNeuralNetwork(x,y)
    def MULT[T2<:Operationable](y:T2) = new MultipliedNeuralNetwork(x,y)
    def REPEAT (n:Int) = new RepeatNeuralNetwork(x, n)
    def TENSOR [T2<:Operationable](y:T2) = // second order tensor only
      new TensorNeuralNetwork(x.outputDimension, y.outputDimension) TIMES (x PLUS y)
    def mTENSOR [T2<:Operationable](y:T2) = // mix first order and second order tensor
      (x TENSOR y) SHARE (x PLUS y)
  } 
}
/** Operationable is a generic trait that supports operations in Workspace */
abstract trait Operationable extends Workspace {
  def inputDimension:Int
  def outputDimension:Int
  
  val key:String = this.hashCode().toString

  def create(): InstanceOfNeuralNetwork
  def toStringGeneric(): String = key +
  	 "[" + inputDimension + "," + outputDimension + "]";
}


class SetOfMemorables extends HashMap[String, Memorable]

/** Memorable is state buffer associate a neural network that keep temporary data which is independent of the parameters of neural network*/
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



/** Class for (template) neural network, where parameters are not instantiated */
abstract class NeuralNetwork (val inputDimension:Int, val outputDimension:Int) extends Operationable{
  type InstanceType <: InstanceOfNeuralNetwork
  def create() : InstanceOfNeuralNetwork 
  override def toString() = "?" + toStringGeneric
}

/** Class for instance of neural network, which can be applied and trained */
abstract class InstanceOfNeuralNetwork (val NN: Operationable) extends Operationable {
  type StructureType <: Operationable
  
  // basic topological structure
  def inputDimension = NN.inputDimension
  def outputDimension= NN.outputDimension
  def create() = this // self reference
  
  /** Instance of neural network are basically function which works on either a single NeuronVector or a set of them as NeuronMatrix */
  def apply (x: NeuronVector, mem:SetOfMemorables) : NeuronVector
  def apply (xs: NeuronMatrix, mem:SetOfMemorables) : NeuronMatrix 

  
  // structure to vectorization functions
  var status:String = ""
  
  /** set parameters of neural network by passing a WeightVector */
  def setWeights(seed:String, w:WeightVector) : Unit = {}
  /** get parameters of neural network as a NeuronVector */
  def getWeights(seed:String): NeuronVector = NullVector 
  /** reset internal parameters (by random) and return a vector */
  def getRandomWeights(seed:String) : NeuronVector = NullVector 
  /** get the dimension of parameters */
  def getDimensionOfWeights(seed: String): Int = 0
  /** get the data of derivative of parameters and return (sample dependent) regularization term */
  def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = 0.0
  
  // dynamic operations
  def init(seed:String, mem: SetOfMemorables) : InstanceOfNeuralNetwork = {this} // default: do nothing
  def allocate(seed:String, mem: SetOfMemorables) : InstanceOfNeuralNetwork = {this} // 
  
  /** backpropagate propagates the partial derivative chain rule */
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


/** basic operation to derive hierarchy structures of two operational neural network */
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

/** {f PLUS g} ([x,y]) := [f(x), g(y)] */
class JointNeuralNetwork [Type1 <: Operationable, Type2 <: Operationable]
		( override val first:Type1, override val second:Type2) 
	extends MergedNeuralNetwork[Type1,Type2](first,second) {
  type InstanceType <: InstanceOfJointNeuralNetwork[Type1,Type2]
  def inputDimension = first.inputDimension + second.inputDimension
  def outputDimension= first.outputDimension+ second.outputDimension 
  def create(): InstanceOfJointNeuralNetwork[Type1, Type2] = new InstanceOfJointNeuralNetwork(this)
  override def toString() = "(" + first.toString + " + " + second.toString + ")"
}

/** f REPEAT n (x) := [f(x), f(x), ... ,f(x)] (repeat n times) */
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

/** {f TIMES g} (x) := f(g(x)) */
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

/** {f SHARE g} (x) := [f(x), g(x)] */
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

/** {f ADD g} (x) := f(x) + g(x) */
class AddedNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val first: Type1, override val second: Type2)
		extends MergedNeuralNetwork[Type1, Type2](first, second) {
  assert (first.inputDimension == second.inputDimension && first.outputDimension == second.outputDimension)
  type InstanceType <: InstanceOfAddedNeuralNetwork[Type1, Type2]
  def inputDimension = first.inputDimension
  def outputDimension = first.outputDimension
  def create(): InstanceOfAddedNeuralNetwork[Type1, Type2] = new InstanceOfAddedNeuralNetwork(this)
  override def toString() = "(" + first.toString +") Join(+) (" + second.toString() + ")"
}

class InstanceOfAddedNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val NN:AddedNeuralNetwork[Type1, Type2])
		extends InstanceOfMergedNeuralNetwork(NN) {
  type StructureType <: AddedNeuralNetwork[Type1, Type2]
  def apply(x:NeuronVector, mem: SetOfMemorables) = {
    val secondVec = secondInstance(x, mem)
    val firstVec  = firstInstance(x, mem)
    firstVec + secondVec
  }
  def apply(x:NeuronMatrix, mem:SetOfMemorables) = {
    val secondMat = secondInstance(x, mem)
    val firstMat = firstInstance(x, mem)
    firstMat + secondMat
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    firstInstance.backpropagate(eta, mem) + secondInstance.backpropagate(eta, mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
	firstInstance.backpropagate(etas, mem) + secondInstance.backpropagate(etas, mem)
  }  
  override def toString() = "(" + firstInstance.toString + ") Join(+) (" + secondInstance.toString + ")"  
}

/** {f Mult g} (x) := f(x) * g(x) */
class MultipliedNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val first: Type1, override val second: Type2)
		extends MergedNeuralNetwork[Type1, Type2](first, second) {
  assert (first.inputDimension == second.inputDimension && first.outputDimension == second.outputDimension)
  type InstanceType <: InstanceOfMultipliedNeuralNetwork[Type1, Type2]
  def inputDimension = first.inputDimension
  def outputDimension = first.outputDimension
  def create(): InstanceOfMultipliedNeuralNetwork[Type1, Type2] = new InstanceOfMultipliedNeuralNetwork(this)
  override def toString() = "(" + first.toString +") Join(+) (" + second.toString() + ")"
}

class InstanceOfMultipliedNeuralNetwork[Type1 <: Operationable, Type2 <: Operationable]
		(override val NN:MultipliedNeuralNetwork[Type1, Type2])
		extends InstanceOfMergedNeuralNetwork(NN) {
  type StructureType <: AddedNeuralNetwork[Type1, Type2]
  override def init(seed:String, mem:SetOfMemorables) = {
    if (!mem.isDefinedAt(key) || mem(key).status != seed) {
      mem. += (key -> new Memorable)
      mem(key).status = seed
      mem(key).numOfMirrors = 1 // find a new instance
      mem(key).mirrorIndex = 0
    }
    else {      
      mem(key).numOfMirrors = mem(key).numOfMirrors + 1
    }
    this
  }
  
  override def allocate(seed:String, mem:SetOfMemorables) ={
    if (mem(key).status == seed) {
      mem(key).outputBuffer= new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).outputBufferM = new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }  
  def apply(x:NeuronVector, mem: SetOfMemorables) = {
    val secondVec = secondInstance(x, mem)
    val firstVec  = firstInstance(x, mem)
    if (mem != null) {
    	mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    	mem(key).outputBuffer(mem(key).mirrorIndex) = firstVec concatenate secondVec;
    }
    firstVec DOT secondVec
  }
  def apply(x:NeuronMatrix, mem:SetOfMemorables) = {
    val secondMat = secondInstance(x, mem)
    val firstMat = firstInstance(x, mem)
    if (mem != null) {
    	mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    	mem(key).outputBufferM(mem(key).mirrorIndex) = firstMat padRow secondMat;
    }    
    firstMat DOT secondMat
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    val (o1, o2) = mem(key).outputBuffer(mem(key).mirrorIndex) splice outputDimension
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    firstInstance.backpropagate(eta DOT o2, mem) + secondInstance.backpropagate(eta DOT o1, mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val (o1, o2) = mem(key).outputBufferM(mem(key).mirrorIndex) spliceRow outputDimension
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors    
	firstInstance.backpropagate(etas DOT o2, mem) + secondInstance.backpropagate(etas DOT o1, mem)
  }  
  override def toString() = "(" + firstInstance.toString + ") Join(+) (" + secondInstance.toString + ")"  
}




