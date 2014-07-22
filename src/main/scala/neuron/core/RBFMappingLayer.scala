package neuron.core
import neuron.math._

class RBFNeuralNetwork (inputDimension: Int, outputDimension: Int /** number of centroids */,
						val rbf: NeuronFunction, val centroids: NeuronMatrix) 
	extends NeuralNetwork (inputDimension, outputDimension) {
  type InstanceType <: InstanceOfRBFNeuralNetwork
  def create(): InstanceOfRBFNeuralNetwork = new InstanceOfRBFNeuralNetwork(this)
}

class InstanceOfRBFNeuralNetwork (override val NN: RBFNeuralNetwork)
	extends InstanceOfNeuralNetwork(NN) {
  type StructureType <: RBFNeuralNetwork
  def apply(x: NeuronVector, mem:SetOfMemorables) = {
    assert(x.length == inputDimension)
    NN.rbf((NN.centroids Minus x).euclideanSqrNormCol)
  }
  
  def apply(x: NeuronMatrix, mem:SetOfMemorables) = {
    assert(x.rows == inputDimension)
    val xx = (NN.centroids :* NN.centroids).sumCol 
    val yy = (x :* x).sumCol
    val xy = (NN.centroids TransMult x ) * (-2)
    NN.rbf(xy Add xx AddTrans yy)
  }
  
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    // to be implemented 
    eta
  }
  
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // to be implemented
    etas
  }
}

class GridRBFNeuralNetwork (inputDimension: Int, val bins: Seq[Int])
	extends NeuralNetwork(inputDimension, bins.foldLeft(1)(_*_)) {
  type InstanceType <: InstanceOfGridRBFNeuralNetwork
  def create(): InstanceOfGridRBFNeuralNetwork = new InstanceOfGridRBFNeuralNetwork(this)
}

class InstanceOfGridRBFNeuralNetwork (override val NN: GridRBFNeuralNetwork) 
	extends InstanceOfNeuralNetwork(NN) {
  type StructureType <: GridRBFNeuralNetwork
  
  val L: NeuronMatrix = new NeuronMatrix(NN.bins.length, outputDimension)
  override def init(seed:String, mem: SetOfMemorables) : InstanceOfNeuralNetwork = {
    for (i<- 0 until outputDimension) {
      var idx_d = i
      for (j<- NN.bins.length-1 to 0 by -1) {
        L(j,i) = (idx_d % NN.bins(j)) * (1.0 / (NN.bins(j)-1))
        idx_d = idx_d / NN.bins(j)
      }
    }
    this
  }
  
  private def gridrbf(y:NeuronVector, x:NeuronVector): Unit = {
    val grids = for (i<- 0 until x.length) yield {
      val f = x(i) * (NN.bins(i)-1)
      val j = f.toInt
      val k = f - j
      (j, k)
    }
    for (i<- 0 until scala.math.pow(2, NN.bins.length).toInt){
      var v = 1.0
      var idx_b = i
      var idx_d = 0
      for (j<- 0 until NN.bins.length) {
        val b = (idx_b % 2)
        idx_d = idx_d * NN.bins(j) + grids(j)._1 + b
        idx_b = idx_b / 2
        v = if (b==0) v*(1-grids(j)._2) else v*grids(j)._2
      }
      y(idx_d) = v
    }    
  }
  def apply(x:NeuronVector, mem:SetOfMemorables) = {
    // assuming the elements in x are in [0,1]
    assert(x.length == inputDimension)
    val y = new NeuronVector(outputDimension)
    gridrbf(y, x)
    y
  }
  def apply(x:NeuronMatrix, mem:SetOfMemorables) = {
    assert(x.rows == inputDimension) 
    val y = new NeuronMatrix(outputDimension, x.cols)
    for (n<-0 until x.cols) {
    	gridrbf(y.colVec(n), x.colVec(n))
    }
    y
  }
  def backpropagate(eta: NeuronVector, mem: SetOfMemorables) = {
    // to be implemented 
    eta
  }
  
  def backpropagate(etas: NeuronMatrix, mem: SetOfMemorables) = {
    // to be implemented
    etas
  }  
}