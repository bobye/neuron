// Copyright: MIT License 2014 Jianbo Ye (jxy198@psu.edu)
package neuron.autoencoder
import scala.concurrent.stm._
import neuron.core._
import neuron.math._

/*******************************************************************************************/
// Encoder and Recursive Neural Network
abstract trait EncoderWorkspace {
  implicit class extractClass [T<:EncodeClass]( x:T) {
    def extract() = x match {
      case _:Encoder => new EncoderNeuralNetwork[T](x)
      case _:InstanceOfEncoder => new InstanceOfEncoderNeuralNetwork[T](x)
      case _ => x // others
    }
  }  
}

class EncoderMemorable extends Memorable {
  var encodeCurrent: NeuronVector = NullVector
  var encodeCurrentM:NeuronMatrix = NullMatrix
}

abstract trait EncodeClass extends Operationable {
  val encodeDimension: Int
}
abstract trait Encoder extends Operationable with EncodeClass{
  type InstanceType <: InstanceOfEncoder
  override def create(): InstanceOfEncoder
} 
abstract trait InstanceOfEncoder extends InstanceOfNeuralNetwork with EncodeClass {
  type StructureType <: Encoder
  def encode(x:NeuronVector, mem:SetOfMemorables): NeuronVector
  def encode(xs:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix 
  //def applyEncodingErr(x:NeuronVector, mem:SetOfMemorables): NeuronVector
  def encodingBP(eta:NeuronVector, mem: SetOfMemorables): NeuronVector 
  def encodingBP(etas:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix
}

// It implicitly requires the dimensional constraints
/*
 * 
abstract trait RecursiveClass extends EncodeClass {
  //val wordLength: Int
  // some bugs here, but works fine for running
  //assert(wordLength == encodeDimension)
  //assert(wordLength*2 == inputDimension)
}

abstract trait RecursiveEncoder extends Operationable with RecursiveClass {
  override def create(): InstanceOfRecursiveEncoder
}
abstract trait InstanceOfRecursiveEncoder extends InstanceOfNeuralNetwork with RecursiveClass 

* 
*/
// Convert Encoder and InstanceOfEncoder to new NeuralNetwork by replacing apply() with encode()
// However, the weights of both are not changed, which means in back propagation 
// only part of weights are got updated
class EncoderNeuralNetwork [T<: Encoder] (val NN: T) 
	extends NeuralNetwork(NN.inputDimension, NN.encodeDimension) {
  type InstanceType <: InstanceOfEncoderNeuralNetwork [NN.InstanceType]
  def create() = new InstanceOfEncoderNeuralNetwork(NN.create())
}

class InstanceOfEncoderNeuralNetwork [T<: InstanceOfEncoder] // T1 and T2 must be compatible
		(val INN: T) 
	extends InstanceOfNeuralNetwork (INN.NN) {
  type StructureType <: EncoderNeuralNetwork[INN.StructureType]  
  
  override val outputDimension = INN.encodeDimension
  def apply (x:NeuronVector, mem:SetOfMemorables) = {
    INN.encode(x, mem)
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    INN.encode(xs, mem)
  }
  override def init(seed:String, mem:SetOfMemorables) = INN.init(seed, mem)
  override def allocate(seed:String, mem:SetOfMemorables) = INN.allocate(seed, mem)
  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    INN.encodingBP(eta, mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
	INN.encodingBP(etas, mem)
  }
  
  override def setWeights(seed:String, w:WeightVector) = INN.setWeights(seed, w)
  override def getWeights(seed:String): NeuronVector = INN.getWeights(seed)
  override def getRandomWeights(seed:String) : NeuronVector = INN.getRandomWeights(seed)
  override def getDimensionOfWeights(seed: String): Int = INN.getDimensionOfWeights(seed)
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = 
    INN.getDerativeOfWeights(seed, dw, numOfSamples)
}

/*******************************************************************************************/
// AutoEncoder
class AutoEncoder (val regCoeff:Double = 0.0,
    			   val encoder: Operationable, val decoder: Operationable,
    			   val distance: DistanceFunction = L2Distance)
	extends SelfTransform (encoder.inputDimension) with Encoder {
  type InstanceType <: InstanceOfAutoEncoder
  assert (encoder.outputDimension == decoder.inputDimension)
  assert (decoder.outputDimension == dimension)
  val encodeDimension = encoder.outputDimension
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class IdentityAutoEncoder (dimension:Int) 
	extends AutoEncoder(0.0, new IdentityTransform(dimension), new IdentityTransform(dimension)) 

class InstanceOfAutoEncoder (override val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with InstanceOfEncoder {
  type Structure <: AutoEncoder
  //protected val inputLayer = new RegularizedLinearNN(NN.dimension, NN.hidden.inputDimension, NN.lambda).create() // can be referenced from ImageAutoEncoder
  //protected val outputLayerLinear = (new RegularizedLinearNN(NN.hidden.outputDimension, NN.dimension, NN.lambda)).create()
  //val outputLayer = (NN.post TIMES outputLayerLinear).create()
  val encodeDimension = NN.encodeDimension
  val encoderInstance = NN.encoder.create()
  val decoderInstance = NN.decoder.create()
  private val main = (decoderInstance ** encoderInstance).create()
  
  private val aeError = Ref(0.0);
  def apply (x:NeuronVector, mem:SetOfMemorables) = { 
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrent = encoderInstance(x, mem) // buffered
    mem(key).outputBuffer(mem(key).mirrorIndex) = 
      decoderInstance(mem(key).asInstanceOf[EncoderMemorable].encodeCurrent, mem)
    
    //mem(key).asInstanceOf[EncoderMemorable].encodingError = 
    //  L2Distance(mem(key).outputBuffer(mem(key).mirrorIndex), x) * NN.regCoeff / mem(key).numOfMirrors    
    if (NN.regCoeff >= 1E-5 && mem(key).mirrorIndex == 0) {
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
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM = encoderInstance(xs, mem) // buffered
    mem(key).outputBufferM(mem(key).mirrorIndex) = 
      decoderInstance(mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM, mem)
      
    if (NN.regCoeff >= 1E-5 && mem(key).mirrorIndex == 0) {
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
    main.init(seed, mem)
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
    main.allocate(seed, mem)
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
  
  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    val z= L2Distance.grad(mem(key).outputBuffer(mem(key).mirrorIndex), 
    					   mem(key).inputBuffer(mem(key).mirrorIndex)) * 
    			(NN.regCoeff/ mem(key).numOfMirrors)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagate(eta + z, mem) - z 
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val z= L2Distance.grad(mem(key).outputBufferM(mem(key).mirrorIndex), 
    					   mem(key).inputBufferM(mem(key).mirrorIndex)) * 
    			(NN.regCoeff/ mem(key).numOfMirrors)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagate(etas + z, mem) - z 
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
    main.setWeights(seed, w) 
  }
  override def getWeights(seed:String): NeuronVector = main.getWeights(seed)
  override def getRandomWeights(seed:String) : NeuronVector = main.getRandomWeights(seed)
  override def getDimensionOfWeights(seed: String): Int = main.getDimensionOfWeights(seed)
  
  // For Auto-Encoder, the encoding error can be used a regularization term in addition
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
       // There is a minor bug here: if hidden layer has sparsity penalty, because we use backpropagation in apply()
       main.getDerativeOfWeights(seed, dw, numOfSamples) + aeError() / numOfSamples
      }
    } else {
      0.0
    }
  }
   
}

  
class LinearAutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction,
    					 override val distance: DistanceFunction = L2Distance) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer), outputLayer, distance)	

class SimpleAutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))			

class RobustAutoEncoder (regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRobustLinearNN = 
	  new RobustLinearNN(dimension, hiddenDimension, 0.3).create(),
	 val outputLayer: InstanceOfLinearNeuralNetwork = 
	  new LinearNeuralNetwork(hiddenDimension, dimension).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))	
	  
class SparseLinearAE (val beta:Double = 0.0, // sparse penalty 
    					   lambda: Double = 0.0, // L2 regularization
    					   regCoeff: Double = 0.0, // autoencoding
    					   val penalty:NeuronFunction = new KL_divergenceFunction(0.01), // average activation
    					   val func: NeuronFunction = SigmoidFunction)
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern	
	extends AutoEncoder(regCoeff,
	    new ChainNeuralNetwork(new SparseSingleLayerNN(hiddenDimension, beta, func, penalty), inputLayer),
	    new RegularizedLinearNN(hiddenDimension, dimension, lambda))

class SparseAutoEncoder (val beta:Double = 0.0,
					  lambda:Double = 0.0,
					  regCoeff: Double = 0.0,
					  val penalty: NeuronFunction = new KL_divergenceFunction(0.01),
					  val func: NeuronFunction = SigmoidFunction)
	(dimension: Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern
	extends AutoEncoder(regCoeff, 
	    new ChainNeuralNetwork(new SparseSingleLayerNN(hiddenDimension, beta, func, penalty), inputLayer),
	    new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), 
	    					   new RegularizedLinearNN(hiddenDimension, dimension, lambda)))
	
class RecursiveLinearAE (func:NeuronFunction = SigmoidFunction) 
	(val wordLength: Int, lambda: Double = 0.0, regCoeff:Double = 0.0) 
	extends LinearAutoEncoder(lambda, regCoeff, func)(wordLength*2, wordLength)() 

class RecursiveSimpleAE (lambda: Double = 0.0, regCoeff:Double = 0.0, func:NeuronFunction =SigmoidFunction) 
	(val wordLength: Int)
	extends SimpleAutoEncoder(lambda, regCoeff, func)(wordLength*2, wordLength)()