// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuron.core

import neuron.math.NeuronVector
import neuron.math.NeuronMatrix

/** (a,b) -> (a tensor b) */
class TensorNeuralNetwork(val firstDimension: Int, val secondDimension: Int) 
	extends NeuralNetwork(firstDimension + secondDimension, firstDimension*secondDimension) {
  type InstanceType = InstanceOfTensorNeuralNetwork
  def create() = new InstanceOfTensorNeuralNetwork(this)
  def createAdHoc() = new InstanceOfTensorNeuralNetworkAdHoc(this)
} 

/** (a,b) -> (a tensor b),a,b */
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
      mem(key).inputBufferM = new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def apply(x:NeuronVector, mem:SetOfMemorables) = {
    if (mem != null) {
    	mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    	mem(key).inputBuffer(mem(key).mirrorIndex) = x;
    }
    val (firstVec, secondVec) = x.splice(NN.firstDimension)
    (firstVec CROSS secondVec).vec() // concatenate firstVec concatenate secondVec
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    if (mem != null) {
      mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
      mem(key).inputBufferM(mem(key).mirrorIndex) = xs;
    }
    
    val (firstVec, secondVec) = xs.spliceRow(NN.firstDimension)
    (firstVec CROSS secondVec).mat()
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    val ord2GradW = eta.asNeuronMatrix(NN.firstDimension, NN.secondDimension) //change ord2Grad -> eta (only)
    val (firstVec, secondVec) = mem(key).inputBuffer(mem(key).mirrorIndex).splice(NN.firstDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (ord2GradW Mult secondVec) concatenate (ord2GradW TransMult firstVec)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    val ord2GradW = etas.asNeuronTensor(NN.firstDimension, NN.secondDimension)
    val (firstVec, secondVec) = mem(key).inputBufferM(mem(key).mirrorIndex).spliceRow(NN.firstDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (ord2GradW Mult secondVec) padRow (ord2GradW TransMult firstVec)
  }
}

/** change to: (a, b) -> \sum a tensor b */
class InstanceOfTensorNeuralNetworkAdHoc (override val NN:TensorNeuralNetwork) 
	extends InstanceOfTensorNeuralNetwork(NN) {
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs;
    val (firstVec, secondVec) = xs.spliceRow(NN.firstDimension)
    firstVec MultTrans secondVec
  }  
  override def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val (firstVec, secondVec) = mem(key).inputBufferM(mem(key).mirrorIndex).spliceRow(NN.firstDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (etas Mult secondVec) padRow (etas TransMult firstVec)
  }
}


/** a -> a tensor a */
class SelfTensorNeuralNetwork(val dimension: Int) 
	extends NeuralNetwork(dimension, dimension*dimension) {
  type InstanceType = InstanceOfSelfTensorNeuralNetwork
  def create() = new InstanceOfSelfTensorNeuralNetwork(this)
  def createAdHoc() = new InstanceOfSelfTensorNeuralNetworkAdHoc(this)
} 


class InstanceOfSelfTensorNeuralNetwork(override val NN:SelfTensorNeuralNetwork) 
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
      mem(key).inputBufferM = new Array[NeuronMatrix] (mem(key).numOfMirrors)
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def apply(x:NeuronVector, mem:SetOfMemorables) = {
    if (mem != null) {
    	mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    	mem(key).inputBuffer(mem(key).mirrorIndex) = x;
    }
    (x CROSS x).vec() // concatenate firstVec concatenate secondVec
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs;
    }
    (xs CROSS xs).mat()
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    val ord2GradW = eta.asNeuronMatrix(NN.dimension, NN.dimension) //change ord2Grad -> eta (only)
    val x = mem(key).inputBuffer(mem(key).mirrorIndex)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (ord2GradW + ord2GradW.transpose) Mult x
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // INCOMPLETE: To be implemented
    val ord2GradW = etas.asNeuronTensor(NN.dimension, NN.dimension)
    val xs = mem(key).inputBufferM(mem(key).mirrorIndex)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    (ord2GradW Mult xs) + (ord2GradW TransMult xs)
  }
}

/** change to: a -> \sum a tensor a */
class InstanceOfSelfTensorNeuralNetworkAdHoc (override val NN:SelfTensorNeuralNetwork) 
	extends InstanceOfSelfTensorNeuralNetwork(NN) {
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs;
    (xs MultTrans xs) / xs.cols
  }  
  override def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val xs = mem(key).inputBufferM(mem(key).mirrorIndex)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    ((etas + etas.transpose) Mult xs) / xs.cols
  }
}
	
