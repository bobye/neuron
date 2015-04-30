package neuron.core
import scala.concurrent.stm._
import neuron.autoencoder.AutoEncoder
import neuron.math._


class TiledWeightLinearNN (inputDimension: Int, outputDimension: Int) 
	extends LinearNeuralNetwork(inputDimension, outputDimension) {
	type InstanceType = InstanceOfTiledWeightLinearNN
	// def create() is identical to LinearNeuralNetwork
    def create(W: NeuronMatrix, dW: Ref[NeuronMatrix], transpose: Boolean = false) = 
      	new InstanceOfTiledWeightLinearNN(this, W: NeuronMatrix, dW: Ref[NeuronMatrix], transpose) 
}

class InstanceOfTiledWeightLinearNN (override val NN: TiledWeightLinearNN, 
    override val W: NeuronMatrix, override val dW: Ref[NeuronMatrix], isTranspose: Boolean = false) 
	extends InstanceOfLinearNeuralNetwork(NN) {
  type StructureType = TiledWeightLinearNN
  
  
  override def setWeights(seed:String, w:WeightVector) : Unit = {
    if (status != seed) {
      status = seed
      w(null, b) // get optimized weights
      //dw(dW, db) // dW and db are distributed states 
      atomic { implicit txn =>
      db():=0.0
      }
    }
  }
  override def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      // W is tiled
      b 
    }else {
      NullVector
    }
  }  
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      // W is tiled
      b := 0.0
      b 
    }else {
      NullVector
    }
  }
  override def getDimensionOfWeights(seed: String): Int = {
    if (status != seed) {
      status = seed
      b.length
    } else {
      0
    }
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dw.get(null, db())      }
    } else {
    }
    0.0
  }
  override def apply (x: NeuronVector, mem:SetOfMemorables) = {
    assert (x.length == inputDimension)
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x
    }
    if (isTranspose)
      (W TransMult x) + b 
    else
      (W Mult x) + b
  }
  override def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    assert (xs.rows == inputDimension)
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs
    }
    if (isTranspose) 
      (W TransMult xs) Add b     
    else
      (W Mult xs) Add b
  }
  override def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    val dWincr = if (isTranspose) mem(key).inputBuffer(mem(key).mirrorIndex) CROSS eta
    			 else eta CROSS mem(key).inputBuffer(mem(key).mirrorIndex)
    atomic { implicit txn =>
    dW() = dW() + dWincr
    db() = db() + eta
    }
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    if (isTranspose)
      W Mult eta // dgemv
    else  
      W TransMult eta
  }
  override def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    val dWincr = if (isTranspose) mem(key).inputBufferM(mem(key).mirrorIndex) MultTrans etas
    			 else etas MultTrans mem(key).inputBufferM(mem(key).mirrorIndex) // dgemm and daxpy
    val dbincr = etas.sumRow()
    atomic { implicit txn =>
    //println(key, mem(key).mirrorIndex, eta.data)
    dW() = dW() + dWincr
    db() = db() + dbincr
    }    
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    if (isTranspose)
      W Mult etas
    else
      W TransMult etas
  }  
}

class TiledWeightSimpleAE (lambda:Double = 0.0,
					  regCoeff: Double = 0.0,
					  val func: NeuronFunction = SigmoidFunction,
					  val outputFunc: NeuronFunction = SigmoidFunction)
	(dimension: Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern
	extends AutoEncoder(regCoeff, 
	    new ChainNeuralNetwork(new SingleLayerNeuralNetwork(hiddenDimension, func), inputLayer),
	    new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, outputFunc),
	    					   new TiledWeightLinearNN(hiddenDimension, dimension).create(inputLayer.W, inputLayer.dW, true)))



class TiledWeightSparseAE (val beta:Double = 0.0,
					  lambda:Double = 0.0,
					  regCoeff: Double = 0.0,
					  val penalty: NeuronFunction = new KL_divergenceFunction(0.01),
					  val func: NeuronFunction = SigmoidFunction,
					  val outputFunc: NeuronFunction = SigmoidFunction)
	(dimension: Int, val hiddenDimension:Int)
	(val inputLayer: InstanceOfRegularizedLinearNN = 
	  new RegularizedLinearNN(dimension, hiddenDimension, lambda).create()) // for visualization concern
	extends AutoEncoder(regCoeff, 
	    new ChainNeuralNetwork(new SparseSingleLayerNN(hiddenDimension, beta, func, penalty), inputLayer),
	    new ChainNeuralNetwork(new SingleLayerNeuralNetwork(dimension, outputFunc),
	    					   new TiledWeightLinearNN(hiddenDimension, dimension).create(inputLayer.W, inputLayer.dW, true)))


