package neuron.classifier

import neuron.math._
import neuron.core._
import scala.concurrent.stm._
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.math._

class NaiveBayesClassifier (val dimension: Int, val numOfLabels: Int, val lambda: Double = 1.0) 
	extends LinearNeuralNetwork(dimension, numOfLabels) {
  type InstanceType <: InstanceOfNaiveBayesClassifier
  override def create() = new InstanceOfNaiveBayesClassifier(this)
}

class InstanceOfNaiveBayesClassifier (NN: NaiveBayesClassifier) 
	extends InstanceOfLinearNeuralNetwork(NN) with Classifier {
  
  val logP = new NeuronMatrix(NN.numOfLabels, NN.dimension)
  val logN = new NeuronMatrix(NN.numOfLabels, NN.dimension)
  def trainOnce() = {
    atomic { implicit txn =>
    	W := ((dW() - NN.lambda) DivElem (db() - NN.lambda * 2.0))
    }
    
    logP := new NeuronMatrix( log(W.data) )
    logN := new NeuronMatrix( log(-W.data + 1.0))

  }
  def classify(x:NeuronVector): Int = {
    val xb = x.filterMap(_>0.5)
    val yb = xb*(-1.0) + 1.0
    (logP * xb + logN * yb).argmax()
  }
  def classify(x:NeuronMatrix): LabelVector = {
    val xb = x.filterMap(_>0.5)
    val yb = xb*(-1.0) + 1.0
    (logP * xb + logN * yb).argmaxCol()
  } 

}