package neuron.autoencoder

import neuron.core._
import neuron.math._

class MLP_AE (val struct: IndexedSeq[Int], 
			  val encodeIndicator: IndexedSeq[Boolean], 
			  lambda:Double = 0.0) (val createLayer: (Int, Int) => Operationable)
			  extends SelfTransform (struct(0)) with Encoder {
  type InstanceType <: InstanceOfMLP_AE
  assert (struct.size == encodeIndicator.size + 1)
  assert (struct(0) == struct.last)
  val sizeOfLayers = struct.size
  val encodeDimension = (0 until struct.size - 1).filter(encodeIndicator(_)).map(struct(_)).sum
  def create(): InstanceOfMLP_AE = new InstanceOfMLP_AE(this)
}

class InstanceOfMLP_AE (override val NN: MLP_AE) extends InstanceOfSelfTransform(NN) with InstanceOfEncoder {
  type StructureType <: MLP_AE
  val encodeDimension = NN.encodeDimension
  
  val main: InstanceOfPerceptron = new Perceptron(NN.struct)(NN.createLayer).create()
  //val encodeList: IndexedSeq[Int] =  (0 until NN.struct.size).filter(NN.encodeIndicator)
  
  def apply (x:NeuronVector, mem:SetOfMemorables) = { 
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBuffer(mem(key).mirrorIndex) = x
    val (o, m) = main.applyLayers(x, mem, NN.encodeIndicator)
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrent = m
       o
    } else {
      main.apply(x, mem)
    }
  }

  def apply (xs:NeuronMatrix, mem:SetOfMemorables) = { 
    if (mem != null) {
    mem(key).mirrorIndex = (mem(key).mirrorIndex - 1 + mem(key).numOfMirrors) % mem(key).numOfMirrors
    mem(key).inputBufferM(mem(key).mirrorIndex) = xs
    val (o, m) = main.applyLayers(xs, mem, NN.encodeIndicator)
    mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM = m          
       o
    } else {
      main.apply(xs, mem)
    }
  }  
  
  def encode(x:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    if (mem != null) {
      apply(x, mem); mem(key).asInstanceOf[EncoderMemorable].encodeCurrent
    } else {
      val (_, m) = main.applyLayers(x, null, NN.encodeIndicator)
      m
    }
  }
  def encode(x:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix = {
    if (mem != null) {
      apply(x, mem); mem(key).asInstanceOf[EncoderMemorable].encodeCurrentM
    } else {
      val (_, m) = main.applyLayers(x, null, NN.encodeIndicator)
      m
    }
  }


  def backpropagate(eta:NeuronVector, mem:SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagate(eta, mem)
  }
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagate(etas, mem) 
  }
  
  def encodingBP(eta:NeuronVector, mem:SetOfMemorables): NeuronVector = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagateLayers(new NeuronVector(outputDimension), eta, mem, NN.encodeIndicator)
  }
  def encodingBP(eta:NeuronMatrix, mem:SetOfMemorables): NeuronMatrix = {
    mem(key).mirrorIndex = (mem(key).mirrorIndex + 1) % mem(key).numOfMirrors
    main.backpropagateLayers(new NeuronMatrix(outputDimension, eta.cols), eta, mem, NN.encodeIndicator)
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
  override def setWeights(seed:String, w:WeightVector): Unit = { 
    main.setWeights(seed, w) 
  }
  override def getWeights(seed:String): NeuronVector = main.getWeights(seed)
  override def getRandomWeights(seed:String) : NeuronVector = main.getRandomWeights(seed)
  override def getDimensionOfWeights(seed: String): Int = main.getDimensionOfWeights(seed)
  
  // For Auto-Encoder, the encoding error can be used a regularization term in addition
  override def getDerativeOfWeights(seed:String, dw:WeightVector, numOfSamples:Int) : Double = {
    if (status != seed) {
      status = seed
      main.getDerativeOfWeights(seed, dw, numOfSamples)
    } else {
      0.0
    }
  }  
}
  
