package neuron.tutorials
import neuron.core._
import neuron.math._
import breeze.stats.distributions._

object RBFMappingLayerTest {
  def main(args: Array[String]): Unit = {
    val x = new NeuronMatrix(3, 100, new Uniform(0,1))
    val nn = new GridRBFNeuralNetwork(3, Seq(16,16,16)).create()
    nn.init("init L", null)
    val y = nn(x,null)
    val z = nn.L * y
    println(z.data)
    println(x.data)
    println(y.sumCol)
  }
}