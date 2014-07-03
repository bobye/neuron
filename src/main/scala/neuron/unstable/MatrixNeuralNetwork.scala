// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updated: April 2014
// Copyright under MIT License
package neuron.unstable
import breeze.stats.distributions._
import scala.concurrent.stm._
import neuron.core._
import neuron.math._

abstract class MatrixNeuralNetwork (val inputX: Int, val inputY:Int, val outputX:Int, val outputY:Int)
	extends NeuralNetwork(inputX*inputY, outputX*outputY){

}

abstract class InstanceOfMatrixNeuralNetwork (override val NN: MatrixNeuralNetwork) 
	extends InstanceOfNeuralNetwork(NN) {
  def applyMatrix(x: NeuronMatrix, mem:SetOfMemorables): NeuronMatrix 
  //def applyTensor(x: NeuronTensor, mem:SetOfMemorables): NeuronTensor
  def backpropagateMatrix(eta: NeuronMatrix, mem:SetOfMemorables): NeuronMatrix
  //def backpropagateTensor(etas: NeuronTensor, mem: SetOfMemorables): NeuronTensor
  
  def apply(x: NeuronVector, mem:SetOfMemorables): NeuronVector = {
    applyMatrix(x.asNeuronMatrix(NN.inputX, NN.inputY), mem).vec()
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    xs
  }  
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    backpropagateMatrix(eta.asNeuronMatrix(NN.outputX, NN.outputY), mem).vec()
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    etas
  }
}

/*
abstract class MatrixSelfTransform (val dimensionX: Int, val dimensionY:Int) 
	extends MatrixNeuralNetwork(dimensionX, dimensionY, dimensionX, dimensionY) 
abstract class InstanceOfMatrixSelfTransform (override val NN: MatrixSelfTransform) extends InstanceOfMatrixNeuralNetwork (NN)
*/

class BiLinearSymmetricNN (val inputTensorDimension: Int, val outputTensorDimension: Int)
	extends MatrixNeuralNetwork (inputTensorDimension, inputTensorDimension, outputTensorDimension, outputTensorDimension) {
  type InstanceType <: InstanceOfBiLinearSymmetricNN
  def create(): InstanceOfBiLinearSymmetricNN = new InstanceOfBiLinearSymmetricNN(this)
}

class InstanceOfBiLinearSymmetricNN (override val NN: BiLinearSymmetricNN) extends InstanceOfMatrixNeuralNetwork(NN) {
  val inputTensorDimension = NN.inputTensorDimension
  val outputTensorDimension= NN.outputTensorDimension
  val W: NeuronMatrix = new NeuronMatrix(outputTensorDimension, inputTensorDimension) 
  val b: NeuronMatrix = new NeuronMatrix (outputTensorDimension, outputTensorDimension)
  val dW = Ref(new NeuronMatrix(outputTensorDimension, inputTensorDimension))
  val db = Ref(new NeuronMatrix (outputTensorDimension, outputTensorDimension))
  
  type StructureType <: BiLinearSymmetricNN
  override def setWeights(seed:String, w:WeightVector) : Unit = {
    if (status != seed) {
      status = seed
      w(W, null) // get optimized weights
      w(b, null)
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
      W.vec() concatenate b.vec()
    } else {
      NullVector
    }    
  }
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      // initialize W: it behaves quite different for Gaussian and Uniform Sampling
      val amplitude:Double = scala.math.sqrt(6.0/(inputTensorDimension + outputTensorDimension + 1.0))
      W := new NeuronMatrix(outputTensorDimension, inputTensorDimension, new Gaussian(0, 1)) 
      W:*= amplitude// randomly set W 
      b.set(0.0)
      W.vec() concatenate b.vec() 
    }else {
      NullVector
    }
  }
  override def getDimensionOfWeights(seed:String): Int = {
    if (status != seed) {
      status = seed
      W.cols * W.rows + b.cols * b.rows
    } else {
      0
    }
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dw.get(dW(), db().vec())//(dW.vec concatenate db) // / numOfMirrors
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
      mem(key).inputBufferM = new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = ""
    } else {}
    this
  }
  
  def applyMatrix(x: NeuronMatrix, mem:SetOfMemorables) = {
    assert (x.rows == inputTensorDimension && x.cols == inputTensorDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = x
    (W * x MultTrans W) + b
  }
  def backpropagateMatrix(eta: NeuronMatrix, mem: SetOfMemorables) = {
    val dWincr = ((eta * W * mem(key).inputBufferM(mem(key).mirrorIndex)) * 2.0)
    atomic { implicit txn =>
    //println(key, mem(key).mirrorIndex, eta.data)
    dW() = dW() +  dWincr
    }
    atomic { implicit txn =>
    db() = db() + eta
    }
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (W TransMult eta) * W // dgemv      
  }
}

class RegularizedBiLinearSymNN (inputTensorDimension: Int, outputTensorDimension: Int, val lambda: Double = 0.0)
	extends BiLinearSymmetricNN(inputTensorDimension, outputTensorDimension) {
  type InstanceType = InstanceOfRegularizedBiLinearSymNN
  override def create(): InstanceOfRegularizedBiLinearSymNN = new InstanceOfRegularizedBiLinearSymNN(this)
}

class InstanceOfRegularizedBiLinearSymNN (override val NN: RegularizedBiLinearSymNN) 
	extends InstanceOfBiLinearSymmetricNN(NN) {
  type StructureType = RegularizedBiLinearSymNN
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dW() = dW() + (W * (NN.lambda * numOfSamples))
      dw.get(dW(), db().vec())//(dW.vec + W.vec * (NN.lambda)) concatenate db
      }
      W.euclideanSqrNorm * (NN.lambda /2)
    } else {
      0.0
    }    
  }
}
