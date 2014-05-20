package neuron.classifier

import neuron.core._
import neuron.math._

trait Classifier {
  def classify(x:NeuronVector): Int 
  def classify(x:NeuronMatrix): LabelVector
}

class LinearClassifier (dimension: Int, numOfLabels: Int, lambda: Double = 0.0) 
	extends RegularizedLinearNN(dimension, numOfLabels, lambda) {
  type InstanceType <: InstanceOfLinearClassifier
  override def create(): InstanceOfLinearClassifier = new InstanceOfLinearClassifier(this)
}

class InstanceOfLinearClassifier(override val NN: LinearClassifier)
	extends InstanceOfRegularizedLinearNN(NN) with Classifier {
  type StructureType <: LinearClassifier
  final var randomGenerator = new scala.util.Random
  def classify(x: NeuronVector): Int = {
    (apply(x, null)).argmax
  }
  
  def classify(x: NeuronMatrix): LabelVector = {
    (apply(x, null)).argmaxCol    
  }
}