package neuralnetwork

// Important change from (a,b) -> (a tensor b, a, b) to (a,b) -> (a tensor b)
class TensorNeuralNetwork(val firstDimension: Int, val secondDimension: Int) 
	extends NeuralNetwork(firstDimension + secondDimension, firstDimension*secondDimension) {
  type InstanceType = InstanceOfTensorNeuralNetwork
  def create() = new InstanceOfTensorNeuralNetwork(this)
} 

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
      mem(key).status = "" // reset status to make sure *Buffer are allocated only once
    } else {} 
    this
  }
  def apply(x:NeuronVector, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x;
    val (firstVec, secondVec) = x.splice(NN.firstDimension)
    (firstVec CROSS secondVec).vec() // concatenate firstVec concatenate secondVec
  }
  def apply(xs:NeuronMatrix, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + mem(key).numOfMirrors - 1) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs;
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
    etas
  }
}
