// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuron.autoencoder
import scala.concurrent.stm._
import neuron.core._
import neuron.math._

// AutoEncoder
class AutoEncoder (val regCoeff:Double = 0.0,
    			   val encoder: Operationable, val decoder: Operationable,
    			   val distance: DistanceFunction = L2Distance)
	extends SelfTransform (encoder.inputDimension) with Encoder {
  type InstanceType <: InstanceOfAutoEncoder
  assert (encoder.outputDimension == decoder.inputDimension)
  assert (decoder.outputDimension == dimension)
  val encodeDimension = encoder.outputDimension
  val copy = (decoder.create() ** encoder.create()).create().copy()
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class IdentityAutoEncoder (dimension:Int) 
	extends AutoEncoder(0.0, new IdentityTransform(dimension), new IdentityTransform(dimension)) 

class InstanceOfAutoEncoder (override val NN: AutoEncoder) 
	extends InstanceOfCopyNeuralNetwork (NN.copy) with InstanceOfEncoder {
  type Structure <: AutoEncoder
  
  val encoderInstance = NN.copy.origin.secondInstance
  val decoderInstance = NN.copy.origin.firstInstance
  val encodeDimension = encoderInstance.outputDimension
  
  private val aeError = Ref(0.0);
  override def apply (x:NeuronVector, mem:SetOfMemorables) = { 
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrent = encoderInstance(x, mem) // buffered
    mem(key).outputBuffer(mem(key).mirrorIndex) = 
      decoderInstance(mem(key).asInstanceOf[EncoderMemorable].encodeCurrent, mem)
    
    //mem(key).asInstanceOf[EncoderMemorable].encodingError = 
    //  L2Distance(mem(key).outputBuffer(mem(key).mirrorIndex), x) * NN.regCoeff / mem(key).numOfMirrors    
    if (NN.regCoeff >= 1E-9 && mem(key).mirrorIndex == 0) {
      // traverse all exists buffers, and compute gradients accordingly
      val regCoeffNorm = NN.regCoeff / mem(key).numOfMirrors
      atomic { implicit txn =>
        aeError() = aeError()  + (0 until mem(key).numOfMirrors).map { i=>
        		  NN.distance(mem(key).outputBuffer(i), mem(key).inputBuffer(i)) * regCoeffNorm
        }.sum
      }
    }

    
    mem(key).outputBuffer(mem(key).mirrorIndex)
    } else {
      decoderInstance(encoderInstance(x, null), null)
    }
  }
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM = encoderInstance(xs, mem) // buffered
    mem(key).outputBufferM(mem(key).mirrorIndex) = 
      decoderInstance(mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM, mem)
      
    if (NN.regCoeff >= 1E-9 && mem(key).mirrorIndex == 0) {
      // traverse all exists buffers, and compute gradients accordingly
      val regCoeffNorm = NN.regCoeff / mem(key).numOfMirrors
      atomic { implicit txn =>
        aeError() = aeError()  + (0 until mem(key).numOfMirrors).map { i=>
        		  NN.distance(mem(key).outputBufferM(i), mem(key).inputBufferM(i)) * regCoeffNorm
        }.sum
      }
    }      
    
    mem(key).outputBufferM(mem(key).mirrorIndex)  
    } else {
      decoderInstance(encoderInstance(xs, null), null)
    }
  }
  
  def encode(x:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    if (mem != null) {
      apply(x, mem); mem(key).asInstanceOf[EncoderMemorable].encodeCurrent
    } else {
      encoderInstance(x, null)
    }
  }
  def encode(xs:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix = {
    if (mem != null) {
      apply(xs, mem); mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM
    } else {
      encoderInstance(xs, null)
    }
  }
  
  override def init(seed:String, mem:SetOfMemorables) = {
    copy.init(seed, mem)
    if (!mem.isDefinedAt(key) || mem(key).status != seed) {
      mem += (key -> new EncoderMemorable)
      mem(key).status = seed
      mem(key).numOfMirrors = 1
      mem(key).mirrorIndex = 0
    } else {
      mem(key).numOfMirrors = mem(key).numOfMirrors + 1
    }
    this
  }
  
  override def allocate(seed:String, mem:SetOfMemorables) : InstanceOfNeuralNetwork = {
    copy.allocate(seed, mem)
    if (mem(key).status == seed) {
      mem(key).inputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).outputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).inputBufferM = new Array[NeuronMatrix](mem(key).numOfMirrors)
      mem(key).outputBufferM= new Array[NeuronMatrix](mem(key).numOfMirrors)
      mem(key).status = ""
    } else {
    }
    this
  }
  
  override def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    if (NN.regCoeff > 1E-9) {
      val z= L2Distance.grad(mem(key).outputBuffer(mem(key).mirrorIndex), 
    					   mem(key).inputBuffer(mem(key).mirrorIndex)) * 
    			(NN.regCoeff/ mem(key).numOfMirrors)
      mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
      copy.backpropagate(eta + z, mem) - z 
    } else {
      mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
      copy.backpropagate(eta, mem)
    }
  }
  override def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    if (NN.regCoeff > 1E-9) {
      val z= L2Distance.grad(mem(key).outputBufferM(mem(key).mirrorIndex), 
    					   mem(key).inputBufferM(mem(key).mirrorIndex)) * 
    			(NN.regCoeff/ mem(key).numOfMirrors)
      mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
      copy.backpropagate(etas + z, mem) - z 
    } else {
      mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
      copy.backpropagate(etas, mem)
    }
  }
  
  def encodingBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    val z= L2Distance.grad(mem(key).outputBuffer(mem(key).mirrorIndex), mem(key).inputBuffer(mem(key).mirrorIndex)) * 
    		(NN.regCoeff/ mem(key).numOfMirrors)
    val z2= decoderInstance.backpropagate(z, mem)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    encoderInstance.backpropagate(z2 + eta, mem) - z
  }
  def encodingBP(etas:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix = {
    val z= L2Distance.grad(mem(key).outputBufferM(mem(key).mirrorIndex), mem(key).inputBufferM(mem(key).mirrorIndex)) * 
    		(NN.regCoeff/ mem(key).numOfMirrors)
    val z2= decoderInstance.backpropagate(z, mem)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    encoderInstance.backpropagate(z2 + etas, mem) - z
  }
  
  def decodingBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    decoderInstance.backpropagate(eta, mem)
  }
  def decodingBP(etas:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix = {
    decoderInstance.backpropagate(etas, mem)
  }

  
  override def setWeights(seed:String, w:WeightVector): Unit = { 
    atomic { implicit txn =>
    	aeError() = 0.0
    }
    copy.setWeights(seed, w) 
  }
  
  // For Auto-Encoder, the encoding error can be used a regularization term in addition
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
       // There is a minor bug here: if hidden layer has sparsity penalty, because we use backpropagation in apply()
       copy.getDerativeOfWeights(seed, dw, numOfSamples) + aeError() / numOfSamples
      }
    } else {
      0.0
    }
  }
   
}


