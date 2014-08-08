package neuron
import neuron.core._
import neuron.math._
import breeze.stats.distributions._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RBFMappingLayerTest extends FunSuite {
  test("test rbf") {
    val x = new NeuronMatrix(3, 100, new Uniform(0,1))
    val nn = new GridRBFNeuralNetwork(3, Seq(16,16,16)).create()
    val y = nn(x,null)
    val z = nn.L * y
    
    assert(AbsFunction(z-x).maxAll < 1E-9)
    assert(AbsFunction(y.sumCol - 1).max < 1E-9)
  }
}