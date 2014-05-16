package neuron.matnn
import scala.concurrent.stm._
import neuron.core._
import neuron.math._


class TiledWeightBiLinearSymNN (inputTensorDimension: Int, outputTensorDimension: Int) 
	extends BiLinearSymmetricNN(inputTensorDimension, outputTensorDimension){
	type InstanceType <: InstanceOfTiledWeightBiLinearSymNN
	def create(W: NeuronMatrix, dW: Ref[NeuronMatrix], transpose: Boolean = false)
		= new InstanceOfTiledWeightBiLinearSymNN(this, W, dW, transpose)
}

class InstanceOfTiledWeightBiLinearSymNN (override val NN: TiledWeightBiLinearSymNN, 
    override val W:NeuronMatrix, override val dW: Ref[NeuronMatrix], isTranspose: Boolean = false)
	extends InstanceOfBiLinearSymmetricNN(NN) {
  type StructureType <: TiledWeightBiLinearSymNN
  override def setWeights(seed:String, w:WeightVector) : Unit = {
    if (status != seed) {
      status = seed
      w(b, null)
      //dw(dW, db) // dW and db are distributed states 
      atomic { implicit txn =>
      db().set(0.0)
      }
    }
  }
  override def getWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      b.vec() 
    }else {
      NullVector
    }
  }  
  override def getRandomWeights(seed:String) : NeuronVector = {
    if (status != seed) {
      status = seed
      b.set(0.0)
      b.vec() 
    }else {
      NullVector
    }
  }
  override def getDimensionOfWeights(seed: String): Int = {
    if (status != seed) {
      status = seed
      b.rows * b.cols
    } else {
      0
    }
  }
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      atomic { implicit txn =>
      dw.get(db(), null)
      }
    } else {
    }
    0.0
  }  
  override def applyMatrix(x: NeuronMatrix, mem:SetOfMemorables) = {
    assert (x.rows == inputTensorDimension && x.cols == inputTensorDimension)
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = x
    if (isTranspose) 
      ((W TransMult x) Mult W) + b
    else
      ((W Mult x) MultTrans W) + b
  }
  override def backpropagateMatrix(eta: NeuronMatrix, mem: SetOfMemorables) = {
    // eta and inputBufferM are symmetric
    val dWincr = if (isTranspose) ((mem(key).inputBufferM(mem(key).mirrorIndex) Mult W Mult eta) Mult 2.0)
    			 else ((eta Mult W Mult mem(key).inputBufferM(mem(key).mirrorIndex)) Mult 2.0)
    atomic { implicit txn =>
    dW() = dW() + dWincr
    }
    atomic { implicit txn =>
    db() = db() + eta
    }
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    if (isTranspose)
      (W Mult eta) MultTrans W
    else  
      (W TransMult eta) Mult W // dgemv      
  }  
}
