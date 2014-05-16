// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuron.core
import breeze.stats.distributions._
import scala.concurrent.stm._
import neuron.math._


/** LinearNeuralNetwork computes a linear transform */
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
  override def getWeights(seed:String): NeuronVector = {
    if (status != seed) {
      status = seed
      W.vec() concatenate b
    } else {
      NullVector
    }
  }
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      // initialize W: it behaves quite different for Gaussian and Uniform Sampling
      val amplitude:Double = scala.math.sqrt(6.0/(outputDimension + inputDimension + 1.0))
      W := new NeuronMatrix(outputDimension, inputDimension, new Gaussian(0, 1)) 
      W:*= amplitude// randomly set W 
      b.set(0.0)
      W.vec() concatenate b 
    }else {
      NullVector
    }
  }
  override def getDimensionOfWeights(seed: String): Int = {
    if (status != seed) {
      status = seed
      W.cols * W.rows + b.length
    } else {
      0
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
  val dW = Ref(new NeuronMatrix(outputDimension, inputDimension))
  val db = Ref(new NeuronVector (outputDimension))
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
    val dWincr = eta CROSS mem(key).inputBuffer(mem(key).mirrorIndex)
    atomic { implicit txn =>
    dW() = dW() + dWincr // dgemm and daxpy
    db() = db() + eta
    }
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    W TransMult eta // dgemv
    
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val dWincr = etas MultTrans mem(key).inputBufferM(mem(key).mirrorIndex) // dgemm and daxpy
    val dbincr = etas.sumRow()
    atomic { implicit txn =>
    //println(key, mem(key).mirrorIndex, eta.data)
    dW() = dW() + dWincr
    db() = db() + dbincr
    }    
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    W TransMult etas
  }
}

/** Equipped LinearNeuralNetwork with weight decay by sigma */
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


