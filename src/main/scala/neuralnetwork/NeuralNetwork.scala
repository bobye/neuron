// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updated: April 2014
// Copyright under MIT License
package neuralnetwork

import breeze.stats.distributions._
import scala.collection.mutable._
import scala.concurrent.stm._
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
  def getRandomWeights(seed:String) : NeuronVector = NullVector
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

// Important change from (a,b) -> (a tensor b, a, b) to (a,b) -> (a tensor b)
class TensorNeuralNetwork(val firstDimension: Int, val secondDimension: Int) 
	extends NeuralNetwork(firstDimension + secondDimension, firstDimension*secondDimension) {
  type InstanceType = InstanceOfTensorNeuralNetwork
  def create() = new InstanceOfTensorNeuralNetwork(this)
} 

class MTensorNeuralNetwork(val firstDimension: Int, val secondDimension: Int) 
	extends ShareNeuralNetwork(new TensorNeuralNetwork(firstDimension, secondDimension), 
							   new IdentityTransform(firstDimension + secondDimension))

class InstanceOfTensorNeuralNetwork(override val NN:TensorNeuralNetwork) 
	extends InstanceOfNeuralNetwork(NN) {
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
      mem(key).inputBuffer= new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def apply(x:NeuronVector, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x;
    val (firstVec, secondVec) = x.splice(NN.firstDimension)
    (firstVec CROSS secondVec).vec() // concatenate firstVec concatenate secondVec
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    xs
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    //val (ord2Grad, ord1Grad) = eta.splice(NN.firstDimension * NN.secondDimension)
    val ord2GradW = eta.asWeight(NN.firstDimension, NN.secondDimension) //change ord2Grad -> eta (only)
    //val (firstOrd1Grad, secondOrd1Grad) = ord1Grad.splice(NN.firstDimension)
    val (firstVec, secondVec) = mem(key).inputBuffer(mem(key).mirrorIndex).splice(NN.firstDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    //val firstOrd2Grad = ord2GradW * secondVec 
    //val secondOrd2Grad = ord2GradW TransMult firstVec
    //(firstOrd2Grad + firstOrd1Grad) concatenate (secondOrd2Grad + secondOrd1Grad)
    (ord2GradW Mult secondVec) concatenate (ord2GradW TransMult firstVec)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    etas
  }
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
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
        status = seed
        firstInstance.getRandomWeights(seed) concatenate secondInstance.getRandomWeights(seed)
      }else {
      NullVector
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
	extends InstanceOfSelfTransform (NN) { 
  type StructureType <: SingleLayerNeuralNetwork
  
  def apply (x: NeuronVector, mem:SetOfMemorables) = {
    assert (x.length == inputDimension)
    //inputBuffer(mirrorIndex) = x
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).gradientBuffer(mem(key).mirrorIndex) = NN.func.grad(x)
    //outputBuffer(mirrorIndex) = NN.func(x)
    
    NN.func(x) // outputBuffer(cIndex)
  }
  def apply (xs:NeuronMatrix, mem:SetOfMemorables) = {
    assert(xs.rows == inputDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).gradientBufferM(mem(key).mirrorIndex) = NN.func.grad(xs)
    NN.func(xs)
  }
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
      mem(key).gradientBuffer= new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).gradientBufferM=new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def backpropagate(eta: NeuronVector, mem:SetOfMemorables) = {
    val cIndex = mem(key).mirrorIndex 
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    eta DOT mem(key).gradientBuffer(cIndex) // there is no penalty for sparsity
  }
  def backpropagate(etas:NeuronMatrix, mem: SetOfMemorables) = {
    val cIndex = mem(key).mirrorIndex
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    etas DOT mem(key).gradientBufferM(cIndex)
  }
}

/** SparseSingleLayer computes average activation and enforce sparsity penalty **/
class SparseSingleLayerNN (override val dimension: Int, 
						   var beta: Double = 0.0,
                           override val func: NeuronFunction = SigmoidFunction /** Pointwise Activation Function **/,
						   val penalty: NeuronFunction = new KL_divergenceFunction(0.04) /** Sparsity Penalty Function **/)
	extends SingleLayerNeuralNetwork (dimension, func) {
  type InstanceType = InstanceOfSparseSingleLayerNN
  override def create (): InstanceOfSparseSingleLayerNN = new InstanceOfSparseSingleLayerNN(this)
} 

class InstanceOfSparseSingleLayerNN (override val NN: SparseSingleLayerNN) 
	extends InstanceOfSingleLayerNeuralNetwork (NN) {
  private var totalUsage: Int = 0 // reset if weights updated
  private val totalUsageOnUpdate = Ref(0)
  override def setWeights(seed: String, w: WeightVector) : Unit = {
    atomic { implicit txn =>
    if (status != seed) {
      status = seed
      totalUsage = totalUsageOnUpdate()
      totalUsageOnUpdate() = 0
      rho := rhoOnUpdate()
      rhoOnUpdate().set(0.0)
    } else {
    }
    }
 }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      if (totalUsage == 0) 0.0 /* use default value */ else {
        //println("s ", NN.penalty(rho / totalUsage).sum * NN.beta)
        NN.penalty(rho / totalUsage).sum * NN.beta
      }
    }else {
      0.0
    }
  }
  override def apply(x: NeuronVector, mem:SetOfMemorables) = {
    val y = super.apply(x, mem)
    atomic { implicit txn =>
    // This part has parallel side effects
    rhoOnUpdate() = rhoOnUpdate() + y; // it still has problems
    totalUsageOnUpdate() = totalUsageOnUpdate() + 1 // for computation of average activation
    }
    y
  }
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    val ys = super.apply(xs, mem)
    atomic { implicit txn =>
    // This part has parallel side effects
    rhoOnUpdate() = rhoOnUpdate() + ys.sumRow(); // it still has problems
    totalUsageOnUpdate() = totalUsageOnUpdate() + 1 // for computation of average activation
    }
    ys  
  }
  private val rho : NeuronVector = new NeuronVector(outputDimension)
  private val rhoOnUpdate = Ref(new NeuronVector(outputDimension))
  override def backpropagate(eta: NeuronVector, mem:SetOfMemorables) = {
    val cIndex = mem(key).mirrorIndex 
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (eta + NN.penalty.grad(rho/totalUsage) * NN.beta) DOT mem(key).gradientBuffer(cIndex)
  }
  override def backpropagate(etas: NeuronMatrix, mem:SetOfMemorables) = {
    val cIndex = mem(key).mirrorIndex 
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (etas Add NN.penalty.grad(rho/totalUsage) * NN.beta) DOT mem(key).gradientBufferM(cIndex)    
  }
}

/** LinearNeuralNetwork computes a linear transform, which is also possible to enforce L1/L2 regularization  **/
class LinearNeuralNetwork (inputDimension: Int, outputDimension: Int) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  type InstanceType <: InstanceOfLinearNeuralNetwork
  def create(): InstanceOfLinearNeuralNetwork = new InstanceOfLinearNeuralNetwork(this)
}
class InstanceOfLinearNeuralNetwork (override val NN: LinearNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) {
  type StructureType <: LinearNeuralNetwork
  override def setWeights(seed:String, w:WeightVector) : Unit = {
    if (status != seed) {
      status = seed
      w(W, b) // get optimized weights
      //dw(dW, db) // dW and db are distributed states 
      atomic { implicit txn =>
      dW().set(0.0) // reset derivative of weights
      db().set(0.0)
      }
    }
  }
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      // initialize W: it behaves quite different for Gaussian and Uniform Sampling
      val amplitude:Double = scala.math.sqrt(6.0/(outputDimension + inputDimension + 1.0))
      W := new NeuronMatrix(outputDimension, inputDimension, new Gaussian(0, 1)) 
      W:*= amplitude// randomly set W 
      
      W.vec() concatenate b 
    }else {
      NullVector
    }
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dw.get(dW(), db())//(dW.vec concatenate db) // / numOfMirrors
      }
    } else {
    }
    0.0
  }

  override def init(seed:String, mem:SetOfMemorables) = {
    if (!mem.isDefinedAt(key) || mem(key).status != seed) {
      mem += (key -> new Memorable)
      mem(key).status = seed
      
      mem(key).numOfMirrors = 1
      mem(key).mirrorIndex  = 0
    }
    else {      
      mem(key).numOfMirrors = mem(key).numOfMirrors + 1
      //println(numOfMirrors)
    }
    this
  }
  override def allocate(seed:String, mem:SetOfMemorables) ={
    if (mem(key).status == seed) {
      mem(key).inputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).inputBufferM = new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = ""
    } else {}
    this
  }  
  val W: NeuronMatrix = new NeuronMatrix(outputDimension, inputDimension) 
  val b: NeuronVector = new NeuronVector (outputDimension)
  protected val dW = Ref(new NeuronMatrix(outputDimension, inputDimension))
  protected val db = Ref(new NeuronVector (outputDimension))
  def apply (x: NeuronVector, mem:SetOfMemorables) = {
    assert (x.length == inputDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x
    (W Mult x) + b 
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    assert (xs.rows == inputDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs
    (W Mult xs) Add b     
  }

  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    /*
    if (mirrorIndex == 0) { // start a new backpropagation
      dW.set(0.0); db.set(0.0)
    }
    * 
    */
    
    atomic { implicit txn =>
    //println(key, mem(key).mirrorIndex, eta.data)
    dW() = dW() + (eta CROSS mem(key).inputBuffer(mem(key).mirrorIndex)) // dgemm and daxpy
    db() = db() + eta
    }
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    W TransMult eta // dgemv
    
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    atomic { implicit txn =>
    //println(key, mem(key).mirrorIndex, eta.data)
    dW() = dW() + (etas MultTrans mem(key).inputBufferM(mem(key).mirrorIndex)) // dgemm and daxpy
    db() = db() + etas.sumRow()
    }    
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    W TransMult etas
  }
}

class RegularizedLinearNN (inputDimension: Int, outputDimension: Int, val lambda: Double = 0.0)
	extends LinearNeuralNetwork (inputDimension, outputDimension) {
  type InstanceType = InstanceOfRegularizedLinearNN
  override def create(): InstanceOfRegularizedLinearNN = new InstanceOfRegularizedLinearNN(this) 
}

class InstanceOfRegularizedLinearNN (override val NN: RegularizedLinearNN) 
	extends InstanceOfLinearNeuralNetwork(NN) {
  type StructureType = RegularizedLinearNN

  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dW() = dW() + (W Mult (NN.lambda * numOfSamples))
      dw.get(dW(), db())//(dW.vec + W.vec * (NN.lambda)) concatenate db
      }
      W.euclideanSqrNorm * (NN.lambda /2)
    } else {
      0.0
    }
    
    
  }
}








