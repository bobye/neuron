package neuron.classifier

import neuron.core._
import neuron.math._

trait Classifier {
  def classify(x:NeuronVector): Int 
  def classify(x:NeuronMatrix): LabelVector
}

class LinearClassifier (dimension: Int, numOfLabels: Int, lambda: Double = 0.0) 
	extends ChainNeuralNetwork(new SingleLayerNeuralNetwork(numOfLabels),
							   new RegularizedLinearNN(dimension, numOfLabels, lambda)) {
  type InstanceType <: InstanceOfLinearClassifier
  override def create(): InstanceOfLinearClassifier = new InstanceOfLinearClassifier(this)
}

class InstanceOfLinearClassifier(override val NN: LinearClassifier)
	extends InstanceOfChainNeuralNetwork(NN) {
  type StructureType <: LinearClassifier
  final var randomGenerator = new scala.util.Random
  def classify(x: NeuronVector): Int = {
    val seed = ((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString
    val mem = new SetOfMemorables
    this.init(seed, mem).allocate(seed, mem)
    (apply(x, mem)).argmax
  }
  
  def classify(x: NeuronMatrix): LabelVector = {
    val seed = ((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString
    val mem = new SetOfMemorables
    this.init(seed, mem).allocate(seed, mem)
    (apply(x, mem)).argmaxCol    
  }
}