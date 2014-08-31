package neuron.autoencoder
import neuron.core._
import neuron.math._

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