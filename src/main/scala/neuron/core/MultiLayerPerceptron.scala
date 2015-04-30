package neuron.core
import neuron.math._

class Perceptron [T<:Operationable] (val struct: IndexedSeq[Int]) (val createLayer: (Int, Int) => T)
	extends NeuralNetwork(struct(0), struct.last) {
	type InstanceType <: InstanceOfPerceptron
	assert (struct.size >= 2)
	val sizeOfLayers: Int = struct.size -1
	
	def create(): InstanceOfPerceptron = sizeOfLayers match {
	  case 1 => {
	    val main = createLayer(struct(0), struct(1)).create().copy()
	    new ImplementationOfSingleLayerPerceptron(main)
	  }
	  case _ => {
	    val child = new Perceptron(struct.dropRight(1))(createLayer).create()
	    val top = createLayer(struct(sizeOfLayers-1), struct(sizeOfLayers)).create()
	    val main = (top ** child).create().copy()
	    new ImplementationOfMultiLayerPerceptron(main)	    	  
	  }
	}
}

trait InstanceOfPerceptron extends InstanceOfNeuralNetwork {
  def applyLayers (x: NeuronVector, mem:SetOfMemorables, index: IndexedSeq[Boolean]): (NeuronVector, NeuronVector)
  def applyLayers (xs: NeuronMatrix, mem:SetOfMemorables, index: IndexedSeq[Boolean]): (NeuronMatrix, NeuronMatrix)
  def backpropagateLayers(eta: NeuronVector, etaB: NeuronVector, mem: SetOfMemorables, index:IndexedSeq[Boolean]): NeuronVector
  def backpropagateLayers(etas: NeuronMatrix, etasB: NeuronMatrix, mem: SetOfMemorables, index:IndexedSeq[Boolean]): NeuronMatrix
}

class ImplementationOfSingleLayerPerceptron(override val NN: CopyNeuralNetwork[InstanceOfNeuralNetwork]) 
	extends InstanceOfCopyNeuralNetwork(NN) with InstanceOfPerceptron {
  def applyLayers (x: NeuronVector, mem:SetOfMemorables, index: IndexedSeq[Boolean]) = {
    assert(index.size == 1)
    val y = apply(x, mem)
    if (index.last) (y, x) else (y, NullVector)
  }
  def applyLayers (x: NeuronMatrix, mem:SetOfMemorables, index: IndexedSeq[Boolean]) = {
    assert(index.size == 1)
    val y = apply(x, mem)
    if (index.last) (y, x) else (y, new NeuronMatrix(0, x.cols))    
  }  
  def backpropagateLayers(eta: NeuronVector, etaB: NeuronVector, mem: SetOfMemorables, index:IndexedSeq[Boolean]) = {
    assert(index.size == 1)
    if (index.last) backpropagate(eta, mem) + etaB 
    else backpropagate(eta, mem)
  }
  def backpropagateLayers(eta: NeuronMatrix, etaB: NeuronMatrix, mem: SetOfMemorables, index:IndexedSeq[Boolean]) = {
    assert(index.size == 1)
    if (index.last) backpropagate(eta, mem) + etaB
    else backpropagate(eta, mem)
  }
  
  
}



class ImplementationOfMultiLayerPerceptron 
		(override val NN: CopyNeuralNetwork[InstanceOfChainNeuralNetwork[InstanceOfNeuralNetwork,InstanceOfPerceptron]])
	extends InstanceOfCopyNeuralNetwork(NN) with InstanceOfPerceptron {

	val child:InstanceOfPerceptron = NN.origin.secondInstance.asInstanceOf[InstanceOfPerceptron]
	val top = NN.origin.firstInstance
	
	def applyLayers(x: NeuronVector, mem:SetOfMemorables, index: IndexedSeq[Boolean]): (NeuronVector, NeuronVector) = {
	  val (o1, l) = child.applyLayers(x, mem, index.dropRight(1))
	  val o2 = top(o1, mem)
	  if (index.last)
	    (o2, o1 concatenate l)
	  else
	    (o2, l)
	}
	def applyLayers(x: NeuronMatrix, mem:SetOfMemorables, index: IndexedSeq[Boolean]): (NeuronMatrix, NeuronMatrix) = {
	  val (o1, l) = child.applyLayers(x, mem, index.dropRight(1))
	  val o2 = top(o1, mem)
	  if (index.last)
	    (o2, o1 padRow l)
	  else
	    (o2, l)
	}	
  def backpropagateLayers(eta: NeuronVector, etaB: NeuronVector, mem: SetOfMemorables, index:IndexedSeq[Boolean]) = {
    if (index.last) {
      val (a, b) = etaB.splice(top.inputDimension)
      child.backpropagateLayers(top.backpropagate(eta, mem) + a, b, mem, index.dropRight(1))
    }
    else
      child.backpropagateLayers(top.backpropagate(eta, mem), etaB, mem, index.dropRight(1))
  }
  def backpropagateLayers(eta: NeuronMatrix, etaB: NeuronMatrix, mem: SetOfMemorables, index:IndexedSeq[Boolean]) = {
    if (index.last) {
      val (a, b) = etaB.spliceRow(top.inputDimension)
      child.backpropagateLayers(top.backpropagate(eta, mem) + a, b, mem, index.dropRight(1))
    }
    else
      child.backpropagateLayers(top.backpropagate(eta, mem), etaB, mem, index.dropRight(1))
  }
}

