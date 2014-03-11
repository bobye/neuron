package neuralnetwork


/********************************************************************************************/
// Encoder and Recursive Neural Network
abstract trait EncodeClass {
  val encodeDimension: Int
}
abstract trait Encoder extends Operationable with EncodeClass{
  type InstanceType <: InstanceOfEncoder
  override def create(): InstanceOfEncoder
} 
abstract trait InstanceOfEncoder extends InstanceOfNeuralNetwork with EncodeClass{
  type StructureType <: Encoder
  val encoder: InstanceOfNeuralNetwork
  def encode(x:NeuronVector): NeuronVector = encoder(x)
}

// It implicitly requires the dimensional constraints
abstract trait RecursiveEncoder extends Encoder {
  override def create(): InstanceOfRecursiveEncoder
}
abstract trait InstanceOfRecursiveEncoder extends InstanceOfEncoder

// Convert Encoder and InstanceOfEncoder to new NeuralNetwork by replacing apply() with encode()
// However, the weights of both are not changed, which means in back propagation 
// only part of weights are got updated
class EncoderNeuralNetwork [T<: Encoder] (val NN: T) extends NeuralNetwork(NN.inputDimension, NN.encodeDimension) {
  type InstanceType <: InstanceOfEncoderNeuralNetwork [NN.InstanceType]
  def create() = new InstanceOfEncoderNeuralNetwork(NN.create())
}

class InstanceOfEncoderNeuralNetwork [T<: InstanceOfEncoder] // T1 and T2 must be compatible
		(val INN: T) 
	extends InstanceOfNeuralNetwork (INN.NN) {
  type StructureType <: EncoderNeuralNetwork[INN.StructureType]  
  
  override val outputDimension = INN.encodeDimension
  def apply (x:NeuronVector) = INN.encode(x)
  override def init(seed:String) = INN.init(seed)
  def backpropagate(eta:NeuronVector) = INN.encoder.backpropagate(eta)
  
  def setWeights(seed:String, w:WeightVector, dw:WeightVector) = INN.setWeights(seed, w, dw)
  def getWeights(seed:String) : NeuronVector = INN.getWeights(seed)
  def getDerativeOfWeights(seed:String) : Double = INN.getDerativeOfWeights(seed)
}

/********************************************************************************************/
// AutoEncoder
class AutoEncoder (override val dimension: Int, val lambda:Double = 0.0, val hidden: NeuralNetwork)
	extends SelfTransform (dimension) with Encoder {
  type InstanceType <: InstanceOfAutoEncoder
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfAutoEncoder = new InstanceOfAutoEncoder(this)
}

class InstanceOfAutoEncoder (override val NN: AutoEncoder) extends InstanceOfSelfTransform (NN) with InstanceOfEncoder {
  type Structure <: AutoEncoder
  protected val inputLayer = new RegularizedLinearNN(NN.dimension, NN.hidden.inputDimension, NN.lambda).create() // can be referenced from ImageAutoEncoder
  private val outputLayer = new RegularizedLinearNN(NN.hidden.outputDimension, NN.dimension, NN.lambda)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val threeLayers = (outputLayer TIMES encoder).create()
  def apply (x:NeuronVector) = threeLayers(x)
  override def init(seed:String) = {threeLayers.init(seed); this}
  override def allocate(seed:String) : InstanceOfNeuralNetwork = {threeLayers.allocate(seed); this}
  def backpropagate(eta:NeuronVector) = threeLayers.backpropagate(eta)
  
  def setWeights(seed:String, w:WeightVector, dw:WeightVector): Unit = { threeLayers.setWeights(seed, w, dw) }
  def getWeights(seed:String) : NeuronVector = threeLayers.getWeights(seed)
  def getDerativeOfWeights(seed:String) : Double = threeLayers.getDerativeOfWeights(seed)
}

class SingleLayerAutoEncoder (val func:NeuronFunction = SigmoidFunction) (dimension:Int, val hiddenDimension:Int, lambda: Double = 0.0) 
	extends AutoEncoder(dimension, lambda, new SingleLayerNeuralNetwork(hiddenDimension, func))

class SparseSingleLayerAE (val beta:Double = 0.0, // sparse penalty 
    					   lambda: Double = 0.0, // L2 regularization
    					   val penalty:NeuronFunction = new KL_divergenceFunction(0.01), // average activation
    					   val func: NeuronFunction = SigmoidFunction)
	(dimension:Int, val hiddenDimension:Int)
	extends AutoEncoder(dimension, lambda, new SparseSingleLayerNN(hiddenDimension, beta, func, penalty))

class RecursiveSingleLayerAE (override val func:NeuronFunction = SigmoidFunction) (val wordLength: Int) 
	extends SingleLayerAutoEncoder(func)(wordLength*2, wordLength) with RecursiveEncoder {
  type Instance <: InstanceOfRecursiveSingleLayerAE
  override def create() : InstanceOfRecursiveSingleLayerAE = new InstanceOfRecursiveSingleLayerAE(this)
}
class InstanceOfRecursiveSingleLayerAE(override val NN:RecursiveSingleLayerAE) 
	extends InstanceOfAutoEncoder(NN) with InstanceOfRecursiveEncoder

/********************************************************************************************/
//Context Aware Auto Encoder
class ContextAwareAutoEncoder(val codeLength: Int, val contextLength: Int, val hidden: NeuralNetwork) 
	extends SelfTransform(codeLength + contextLength) with Encoder {
  type Instance <: InstanceOfContextAwareAutoEncoder
  val encodeDimension = hidden.outputDimension
  def create (): InstanceOfContextAwareAutoEncoder = new InstanceOfContextAwareAutoEncoder(this)
}
class InstanceOfContextAwareAutoEncoder(override val NN:ContextAwareAutoEncoder) 
	extends InstanceOfSelfTransform(NN) with InstanceOfEncoder {
  type Structure <: ContextAwareAutoEncoder
  private val inputLayer = new LinearNeuralNetwork(NN.codeLength +NN.contextLength, NN.hidden.inputDimension)
  private val finalLayer = new LinearNeuralNetwork(NN.hidden.outputDimension + NN.contextLength, NN.codeLength)
  val encodeDimension = NN.hidden.outputDimension
  val encoder = (NN.hidden TIMES inputLayer).create()
  private val topLayer = finalLayer.create()
  def apply (x:NeuronVector) = {
    var (_, context) = x.splice(NN.codeLength)
    topLayer(encoder(x) concatenate context) concatenate context
  } 
  override def init(seed:String) = {
    encoder.init(seed)
    topLayer.init(seed)
    this
  }
  override def allocate(seed:String) : InstanceOfNeuralNetwork = {
    encoder.allocate(seed);
    topLayer.allocate(seed)
    this
  }
  def backpropagate(eta:NeuronVector) = {
    var (eta_31, _) = eta.splice(NN.codeLength)
    var (eta_21, _) = topLayer.backpropagate(eta_31).splice(NN.hidden.outputDimension)
    encoder.backpropagate(eta_21)
  }
  
  def setWeights(seed:String, w:WeightVector, dw:WeightVector): Unit = {
    if (status != seed) {
    	encoder.setWeights(seed, w, dw) 
    	topLayer.setWeights(seed, w, dw)
    } else {
    }
  }
  def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      encoder.getWeights(seed) concatenate topLayer.getWeights(seed) 
    } else NullVector
  }
  def getDerativeOfWeights(seed:String) : Double = {
    if (status != seed) {
      status = seed
      encoder.getDerativeOfWeights(seed) +
      topLayer.getDerativeOfWeights(seed)  
    } else {
      0.0
    }
  }
  
}

class SingleLayerCAE(val func: NeuronFunction = SigmoidFunction) 
	(override val codeLength: Int, override val contextLength:Int, val hiddenDimension: Int)
	extends ContextAwareAutoEncoder(codeLength, contextLength, new SingleLayerNeuralNetwork(hiddenDimension, func))

class RecursiveSingleLayerCAE (override val func: NeuronFunction = SigmoidFunction) 
	(val wordLength:Int, override val contextLength: Int)
	extends SingleLayerCAE(func)(wordLength*2, contextLength, wordLength) with RecursiveEncoder {
  type Instance <: InstanceOfRecursiveSingleLayerCAE
  override def create(): InstanceOfRecursiveSingleLayerCAE = new InstanceOfRecursiveSingleLayerCAE(this)
}
class InstanceOfRecursiveSingleLayerCAE (override val NN: RecursiveSingleLayerCAE)
	extends InstanceOfContextAwareAutoEncoder(NN) with InstanceOfRecursiveEncoder

