package neuron.misc

import neuron.core._
import neuron.math._

/**
 * @author bobye
 */
class FractalNeuralNetwork (val depth: Int, val a: Operationable) 
  extends Operationable {
  val b = a.create()
  val inputDimension = (scala.math.pow(2, depth) * a.inputDimension).toInt
  val outputDimension = a.outputDimension  
  def create() = if (depth == 0) {
    a.create()
  } else {
    ((a + b) ** new FractalNeuralNetwork(depth-1, b ++ a)).create()
  }
}