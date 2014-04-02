package neuralnetwork

import scala.concurrent.stm._

/********************************************************************************************/
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
  //val encoder: InstanceOfNeuralNetwork
  def encode(x:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    apply(x, mem); mem(key).asInstanceOf[EncoderMemorable].encodeCurrent
  }
  //def applyEncodingErr(x:NeuronVector, mem:SetOfMemorables): NeuronVector
  def encodingBP(eta:NeuronVector, mem: SetOfMemorables): NeuronVector 
}

// It implicitly requires the dimensional constraints
abstract trait RecursiveClass extends EncodeClass {
  val wordLength: Int
}
abstract trait RecursiveEncoder extends Operationable with RecursiveClass {
  override def create(): InstanceOfRecursiveEncoder
}
abstract trait InstanceOfRecursiveEncoder extends InstanceOfNeuralNetwork with RecursiveClass 

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
  override def init(seed:String, mem:SetOfMemorables) = INN.init(seed, mem)
  override def allocate(seed:String, mem:SetOfMemorables) = INN.allocate(seed, mem)
  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    INN.encodingBP(eta, mem)
  }
  
  override def setWeights(seed:String, w:WeightVector) = INN.setWeights(seed, w)
  override def getRandomWeights(seed:String) : NeuronVector = INN.getRandomWeights(seed)
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = 
    INN.getDerativeOfWeights(seed, dw, numOfSamples)
}

/********************************************************************************************/
// AutoEncoder
class AutoEncoder (override val dimension: Int, val regCoeff:Double = 0.0,
    			   val encoder: Operationable, val decoder: Operationable)
	extends SelfTransform (dimension) with Encoder {
  type InstanceType <: InstanceOfAutoEncoder
  assert (encoder.inputDimension == dimension)
  assert (encoder.outputDimension == decoder.inputDimension)
  assert (decoder.outputDimension == dimension)
  val encodeDimension = encoder.outputDimension
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class InstanceOfAutoEncoder (override val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with InstanceOfEncoder {
  type Structure <: AutoEncoder
  //protected val inputLayer = new RegularizedLinearNN(NN.dimension, NN.hidden.inputDimension, NN.lambda).create() // can be referenced from ImageAutoEncoder
  //protected val outputLayerLinear = (new RegularizedLinearNN(NN.hidden.outputDimension, NN.dimension, NN.lambda)).create()
  //val outputLayer = (NN.post TIMES outputLayerLinear).create()
  val encodeDimension = NN.encodeDimension
  val encoderInstance = NN.encoder.create()
  val decoderInstance = NN.decoder.create()
  private val main = (decoderInstance TIMES encoderInstance).create()
  
  private val aeError = Ref(0.0);
  def apply (x:NeuronVector, mem:SetOfMemorables) = { 
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
        		  L2Distance(mem(key).outputBuffer(i), mem(key).inputBuffer(i)) * regCoeffNorm
        }.sum
      }
    }

    
    mem(key).outputBuffer(mem(key).mirrorIndex)    
  }
  
  
  override def init(seed:String, mem:SetOfMemorables) = {
    if (!mem.isDefinedAt(key) || mem(key).status != seed) {
      mem += (key -> new EncoderMemorable)
      mem(key).status = seed
      mem(key).numOfMirrors = 1
      mem(key).mirrorIndex = 0
      main.init(seed, mem) 
    } else {
      main.init(seed, mem)
      mem(key).numOfMirrors = mem(key).numOfMirrors + 1
    }
    this
  }
  
  override def allocate(seed:String, mem:SetOfMemorables) : InstanceOfNeuralNetwork = {
    if (mem(key).status == seed) {
      main.allocate(seed, mem);
      mem(key).inputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).outputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).status = ""
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
  
  def encodingBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    val z= L2Distance.grad(mem(key).outputBuffer(mem(key).mirrorIndex), mem(key).inputBuffer(mem(key).mirrorIndex)) * 
    		(NN.regCoeff/ mem(key).numOfMirrors)
    val z2= decoderInstance.backpropagate(z, mem)
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    encoderInstance.backpropagate(z2 + eta, mem) - z
  }
  
  def decodingBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    decoderInstance.backpropagate(eta, mem)
  }

  
  override def setWeights(seed:String, w:WeightVector): Unit = { 
    atomic { implicit txn =>
    	aeError() = 0.0
    }
    main.setWeights(seed, w) 
  }
  override def getRandomWeights(seed:String) : NeuronVector = main.getRandomWeights(seed)
  
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

object AutoEncoderCases extends Workspace {
class LinearAutoEncoder (val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int, lambda: Double = 0.0, regCoeff: Double = 0.0) 
	extends AutoEncoder(dimension, regCoeff, 
			new SingleLayerNeuralNetwork(hiddenDimension, func) TIMES new RegularizedLinearNN(dimension, hiddenDimension, lambda),
			new RegularizedLinearNN(hiddenDimension, dimension, lambda))

class SimpleAutoEncoder (val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int, lambda: Double = 0.0, regCoeff: Double = 0.0)
	extends AutoEncoder(dimension, regCoeff, 
			new SingleLayerNeuralNetwork(hiddenDimension, func) TIMES new RegularizedLinearNN(dimension, hiddenDimension, lambda),
			new SingleLayerNeuralNetwork(dimension, func) TIMES new RegularizedLinearNN(hiddenDimension, dimension, lambda))
			

	  
class SparseLinearAE (val beta:Double = 0.0, // sparse penalty 
    					   lambda: Double = 0.0, // L2 regularization
    					   regCoeff: Double = 0.0, // autoencoding
    					   val penalty:NeuronFunction = new KL_divergenceFunction(0.01), // average activation
    					   val func: NeuronFunction = SigmoidFunction)
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern	
	extends AutoEncoder(dimension, regCoeff,
	    new SparseSingleLayerNN(hiddenDimension, beta, func, penalty) TIMES inputLayer,
	    new RegularizedLinearNN(hiddenDimension, dimension, lambda))

class SparseAutoEncoder (val beta:Double = 0.0,
					  lambda:Double = 0.0,
					  regCoeff: Double = 0.0,
					  val penalty: NeuronFunction = new KL_divergenceFunction(0.01),
					  val func: NeuronFunction = SigmoidFunction,
					  val outputFunc: NeuronFunction = SigmoidFunction)
	(dimension: Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern
	extends AutoEncoder(dimension, regCoeff, 
	    new SparseSingleLayerNN(hiddenDimension, beta, func, penalty) TIMES inputLayer,
	    new SingleLayerNeuralNetwork(dimension, func) TIMES new RegularizedLinearNN(hiddenDimension, dimension, lambda))
	
class RecursiveLinearAE (func:NeuronFunction = SigmoidFunction) 
	(val wordLength: Int, lambda: Double = 0.0, regCoeff:Double = 0.0) 
	extends LinearAutoEncoder(func)(wordLength*2, wordLength, lambda, regCoeff) with RecursiveEncoder {
  type Instance <: InstanceOfRecursiveLinearAE
  override def create() : InstanceOfRecursiveLinearAE = new InstanceOfRecursiveLinearAE(this)
}
class InstanceOfRecursiveLinearAE(override val NN:RecursiveLinearAE) 
	extends InstanceOfAutoEncoder(NN) with InstanceOfRecursiveEncoder {
  val wordLength = NN.wordLength
}

class RecursiveSimpleAE (func:NeuronFunction =SigmoidFunction) 
	(val wordLength: Int, lambda: Double = 0.0, regCoeff:Double = 0.0)
	extends SimpleAutoEncoder(func)(wordLength*2, wordLength, lambda, regCoeff) with RecursiveEncoder {
  type Instance <: InstanceOfRecursiveSimpleAE
  override def create() : InstanceOfRecursiveSimpleAE = new InstanceOfRecursiveSimpleAE(this)
}
class InstanceOfRecursiveSimpleAE(override val NN:RecursiveSimpleAE)
	extends InstanceOfAutoEncoder(NN) with InstanceOfRecursiveEncoder {
  val wordLength = NN.wordLength
}

}
     			 
/********************************************************************************************/
// Context Aware Auto Encoder (NOT YET TESTED!)
// The following section is far from complete! Don't use it.
/* 
abstract trait ContextAwareClass extends Operationable {
  val contextLength: Int
}
abstract trait ContextAwareEncoder extends Encoder with ContextAwareClass {
  override def create(): InstanceOfContextAwareEncoder
}
abstract trait InstanceOfContextAwareEncoder extends InstanceOfEncoder with ContextAwareClass

class ContextAwareAutoEncoder(val codeLength: Int, val contextLength: Int, 
    val hidden: NeuralNetwork, val post: SelfTransform) 
	extends SelfTransform(codeLength + contextLength) with ContextAwareEncoder {
  type Instance <: InstanceOfContextAwareAutoEncoder
  assert (post.dimension == codeLength)
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfContextAwareAutoEncoder = new InstanceOfContextAwareAutoEncoder(this)
}
class InstanceOfContextAwareAutoEncoder(override val NN:ContextAwareAutoEncoder) 
	extends InstanceOfSelfTransform(NN) with InstanceOfContextAwareEncoder {
  type Structure <: ContextAwareAutoEncoder
  val contextLength = NN.contextLength
  private val inputLayer = new LinearNeuralNetwork(NN.codeLength +NN.contextLength, NN.hidden.inputDimension).create()
  private val outputLayerLinear = new LinearNeuralNetwork(NN.hidden.outputDimension + NN.contextLength, NN.codeLength).create()
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val outputLayer = (NN.post TIMES outputLayerLinear).create()
  
  
  def apply (x:NeuronVector, mem:SetOfMemorables) = {
    var (_, context) = x.splice(NN.codeLength)
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrent = encoder(x, mem)
    val cIndex = mem(key).mirrorIndex
    mem(key).outputBuffer(mem(key).mirrorIndex) = 
      outputLayer(mem(key).asInstanceOf[EncoderMemorable].encodeCurrent 
          concatenate context, mem) concatenate context
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors 
    mem(key).outputBuffer(cIndex)
  } 
  
  override def init(seed:String, mem:SetOfMemorables) = {
    if (!mem.isDefinedAt(key) || mem(key).status != seed) {
      mem += (key -> new EncoderMemorable)
      mem(key).status = seed
      encoder.init(seed, mem)
      outputLayer.init(seed, mem)
      mem(key).mirrorIndex = 0
    }
    this
  }
  override def allocate(seed:String, mem:SetOfMemorables) : InstanceOfNeuralNetwork = {
    if (mem(key).status == seed) {
      encoder.allocate(seed, mem)
      mem(key).inputBuffer = mem(inputLayer.key).inputBuffer
      mem(key).numOfMirrors = mem(outputLayerLinear.key).numOfMirrors
      mem(key).outputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      outputLayer.allocate(seed, mem)
      mem(key).status = ""
    }  
    this
  }
  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    var (eta_31, _) = eta.splice(NN.codeLength)
    var (eta_21, _) = outputLayer.backpropagate(eta_31, mem).splice(NN.hidden.outputDimension)
    encoder.backpropagate(eta_21, mem)
  }
  
  def encodingErrBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {NullVector} // to be implemented
  
  override def setWeights(seed:String, w:WeightVector): Unit = {
    if (status != seed) {
    	encoder.setWeights(seed, w) 
    	outputLayer.setWeights(seed, w)
    } else {
    }
  }
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      encoder.getRandomWeights(seed) concatenate outputLayer.getRandomWeights(seed) 
    } else NullVector
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      encoder.getDerativeOfWeights(seed, dw, numOfSamples) +
      outputLayer.getDerativeOfWeights(seed, dw, numOfSamples)  
    } else {
      0.0
    }
  }
  
}

class SingleLayerCAE(val func: NeuronFunction = SigmoidFunction) 
	(override val codeLength: Int, override val contextLength:Int, val hiddenDimension: Int)
	extends ContextAwareAutoEncoder(codeLength, contextLength, 
	    new SingleLayerNeuralNetwork(hiddenDimension, func),
	    new SingleLayerNeuralNetwork(codeLength, SigmoidFunction)) with ContextAwareEncoder

abstract trait CREncodeClass extends EncodeClass with RecursiveClass with ContextAwareClass
abstract trait CREncoder extends Encoder with RecursiveEncoder with ContextAwareEncoder {
  type InstanceType <: InstanceOfCREncoder
  override def create(): InstanceOfCREncoder
}
abstract trait InstanceOfCREncoder extends InstanceOfEncoder with InstanceOfRecursiveEncoder with InstanceOfContextAwareEncoder {
  type StructureType <: CREncoder
}

class RecursiveSingleLayerCAE (override val func: NeuronFunction = SigmoidFunction) 
	(val wordLength:Int, override val contextLength: Int)
	extends SingleLayerCAE(func)(wordLength*2, contextLength, wordLength) 
	with CREncoder {
  type Instance <: InstanceOfRecursiveSingleLayerCAE
  override def create(): InstanceOfRecursiveSingleLayerCAE = new InstanceOfRecursiveSingleLayerCAE(this)
}
class InstanceOfRecursiveSingleLayerCAE (override val NN: RecursiveSingleLayerCAE)
	extends InstanceOfContextAwareAutoEncoder(NN) 
	with InstanceOfCREncoder{
  val wordLength = NN.wordLength
}

*/
