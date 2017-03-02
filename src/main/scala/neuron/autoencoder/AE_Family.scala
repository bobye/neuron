package neuron.autoencoder
import neuron.core._
import neuron.math._

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


class ReLUAutoEncoder (lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction,
    					 override val distance: DistanceFunction = L2Distance) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer), 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, ReLUFunction), outputLayer), distance)	

class SimpleAutoEncoder (var lambda: Double = 0.0, 
     					 regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, func), outputLayer))	{
  override def create(): InstanceOfSimpleAutoEncoder = new InstanceOfSimpleAutoEncoder(this)
}

class InstanceOfSimpleAutoEncoder (override val NN: SimpleAutoEncoder)
	extends InstanceOfAutoEncoder(NN) {
  def setLambda(lbd: Double): Unit = {
    NN.inputLayer.setLambda(lbd)
    NN.outputLayer.setLambda(lbd)
  }
}

class ContractiveAutoEncoder (lambda: Double = 0.0, override val regCoeff: Double = 0.0) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create(),
	 val outputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
	extends AutoEncoder(regCoeff, 
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension), inputLayer),
			new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension), outputLayer))	{
  override def create(): InstanceOfContractiveAutoEncoder = new InstanceOfContractiveAutoEncoder(this)
}

class InstanceOfContractiveAutoEncoder(override val NN: ContractiveAutoEncoder) 
	extends InstanceOfAutoEncoder(NN) {
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    if (mem != null) {
      val mem_enc = mem(key).asInstanceOf[EncoderMemorable]
      mem_enc.mirrorIndex = (mem_enc.mirrorIndex - 1 + mem_enc.numOfMirrors) % mem_enc.numOfMirrors
      mem_enc.inputBufferM(mem_enc.mirrorIndex) = xs
      mem_enc.encodeCurrentM = encoderInstance(xs, mem) // buffered
      mem_enc.outputBufferM(mem_enc.mirrorIndex) = 
      decoderInstance(mem_enc.encodeCurrentM, mem)
      if (NN.regCoeff > 1E-9) {
    	  mem_enc.buffer0(mem_enc.mirrorIndex) = (mem_enc.encodeCurrentM :* (mem_enc.encodeCurrentM - 1)).sumRow() * (- NN.regCoeff / xs.cols)
    	  mem_enc.buffer0M(mem_enc.mirrorIndex) = mem_enc.encodeCurrentM
    	  if (mem_enc.mirrorIndex == 0) {
    	    mem_enc.buffer1 = (NN.inputLayer.W :* NN.inputLayer.W).sumRow()
    	    mem_enc.buffer2 = (NN.outputLayer.W :* NN.outputLayer.W).sumCol()
    	  }
      }
      mem_enc.outputBufferM(mem_enc.mirrorIndex)
    } else {
      decoderInstance(encoderInstance(xs, null), null)
    }
  }
  override def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    import scala.concurrent.stm._
    val mem_enc = mem(key).asInstanceOf[EncoderMemorable]
    if (NN.regCoeff > 1E-9) {
      mem_enc.mirrorIndex = (mem_enc.mirrorIndex + 1) % mem_enc.numOfMirrors
      atomic { implicit txn =>
      	NN.inputLayer.dW() = NN.inputLayer.dW() + (NN.inputLayer.W MultElem (mem_enc.buffer0(mem_enc.mirrorIndex) :* mem_enc.buffer2))
      	NN.outputLayer.dW() = NN.outputLayer.dW() + (NN.outputLayer.W MultElemTrans (mem_enc.buffer0(mem_enc.mirrorIndex) :* mem_enc.buffer1))
      }
      val z = (mem_enc.buffer0M(mem_enc.mirrorIndex) - 0.5) MultElem (mem_enc.buffer1 :* mem_enc.buffer2 * (-NN.regCoeff / etas.cols))
      encoderInstance.backpropagate(decoderInstance.backpropagate(etas, mem) + z, mem)
    } else {
      mem_enc.mirrorIndex = (mem_enc.mirrorIndex + 1) % mem_enc.numOfMirrors
      copy.backpropagate(etas, mem)
    }
  }
  override def allocate(seed:String, mem:SetOfMemorables) : InstanceOfNeuralNetwork = {
    copy.allocate(seed, mem)
    if (mem(key).status == seed) {
      mem(key).inputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).outputBuffer = new Array[NeuronVector] (mem(key).numOfMirrors)
      mem(key).inputBufferM = new Array[NeuronMatrix](mem(key).numOfMirrors)
      mem(key).outputBufferM= new Array[NeuronMatrix](mem(key).numOfMirrors)
      mem(key).asInstanceOf[EncoderMemorable].buffer0 = new Array[NeuronVector](mem(key).numOfMirrors)
      mem(key).asInstanceOf[EncoderMemorable].buffer0M = new Array[NeuronMatrix](mem(key).numOfMirrors)
      mem(key).status = ""
    } else {
    }
    this
  }  
}

class RobustAutoEncoder (lambda: Double = 0.0, regCoeff: Double = 0.0, 
    					 val func:NeuronFunction = SigmoidFunction) 
	(dimension:Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRobustLinearNN = 
	  new RobustLinearNN(dimension, hiddenDimension, 0.1, lambda).create(),
	 val outputLayer: InstanceOfLinearNeuralNetwork = 
	  new RegularizedLinearNN(hiddenDimension, dimension, lambda).create())
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