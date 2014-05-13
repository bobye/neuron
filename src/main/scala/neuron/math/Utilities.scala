// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updated: April 2014
// Copyright under MIT License
package neuron.math

/********************************************************************************************/
// Numerical classes and their operations : interface to breeze
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats.distributions._
import neuron.core._
//import breeze.math._


class NeuronVector (val data: DenseVector[Double]) {
  def length = data.length
  def this(n:Int) = this(DenseVector.zeros[Double] (n))
  def this(n:Int, rand: Rand[Double]) = this(DenseVector.rand(n, rand)) // uniform sampling, might not be a good default choice
  def concatenate (that: NeuronVector) : NeuronVector = new NeuronVector(DenseVector.vertcat(this.data, that.data))
  def splice(num: Int) : (NeuronVector, NeuronVector) = (new NeuronVector(this.data(0 until num)), new NeuronVector(this.data(num to -1)))

  def -(that:NeuronVector): NeuronVector = new NeuronVector(this.data - that.data)
  def +(that:NeuronVector): NeuronVector = new NeuronVector(this.data + that.data)
  def +(x:Double) : NeuronVector = new NeuronVector(this.data + x)
  def -(x:Double) : NeuronVector = new NeuronVector(this.data - x)
  def *(x:Double) : NeuronVector = new NeuronVector(this.data * x)
  def /(x:Double) : NeuronVector = new NeuronVector(this.data / x)
  def :=(that: NeuronVector): Unit = {this.data := that.data; }
  def +=(that: NeuronVector): Unit = {this.data :+= that.data; }
  def :*=(x:Double): Unit = {this.data :*= x}
  def :/=(x:Double): Unit = {this.data :/= x}
  def euclideanSqrNorm = {val z = norm(data); z*z}
  def DOT(that: NeuronVector): NeuronVector = new NeuronVector(this.data :* that.data)
  def CROSS (that: NeuronVector): NeuronMatrix = new NeuronMatrix(this.data.asDenseMatrix.t * that.data.asDenseMatrix)
  
  def set(x:Double) : Unit = {data:=x; }
  def copy(): NeuronVector = new NeuronVector(data.copy)
  def sum(): Double = data.sum
  def asNeuronMatrix(rows:Int, cols:Int): NeuronMatrix = new NeuronMatrix (data.asDenseMatrix.reshape(rows, cols)) 
  def last(): Double = data(data.length)
  def append(last: Double): NeuronVector = new NeuronVector(DenseVector.vertcat(data, DenseVector(last)) )
  def normalized(): NeuronVector = new NeuronVector(data/norm(data))
  override def toString() = data.data.mkString("\t")
  def toWeightVector(): WeightVector = new WeightVector(data)
  def importFromFile(filename: String): Unit = {
    val source = scala.io.Source.fromFile(filename)
    val dataBlock = source.mkString.split("\\s+").map(_.toDouble)
    source.close()
    assert(dataBlock.length == length)
    data := new DenseVector(dataBlock)
  }
}
class NeuronMatrix (val data:DenseMatrix[Double]){
  def rows = data.rows
  def cols = data.cols
  def this(rows:Int, cols:Int) = this(DenseMatrix.zeros[Double](rows,cols))
  def this(rows:Int, cols:Int, rand: Rand[Double]) = this(DenseMatrix.rand(rows, cols, rand)) // will be fixed in next release
  def +(that: NeuronMatrix): NeuronMatrix = new NeuronMatrix(this.data + that.data)
  def -(that: NeuronMatrix): NeuronMatrix = new NeuronMatrix(this.data - that.data)  
  def Add(that: NeuronVector): NeuronMatrix = new NeuronMatrix(this.data(::, *) + that.data)
  def AddTrans(that:NeuronVector): NeuronMatrix = new NeuronMatrix(this.data(*, ::) + that.data)
  def MultElem(that: NeuronVector): NeuronMatrix = new NeuronMatrix(this.data(*,::) :* that.data)
  def MultElemTrans(that:NeuronVector): NeuronMatrix = new NeuronMatrix(this.data(::,*) :* that.data)
  def Mult(x:NeuronVector) = new NeuronVector(data * x.data) //this * x
  def TransMult(x:NeuronVector): NeuronVector = new NeuronVector(this.data.t * x.data)
  def Mult(x:NeuronMatrix): NeuronMatrix = new NeuronMatrix(this.data * x.data)
  def TransMult(x:NeuronMatrix) = new NeuronMatrix(this.data.t * x.data)
  def MultTrans(x:NeuronMatrix) = new NeuronMatrix(this.data * x.data.t)  
  def Mult(x:Double): NeuronMatrix = new NeuronMatrix(this.data * x)
  def +(x:Double): NeuronMatrix = new NeuronMatrix(this.data + x)
  def -(x:Double): NeuronMatrix = new NeuronMatrix(this.data - x)
  def /(x:Double) : NeuronMatrix = new NeuronMatrix(this.data / x)
  def :=(that:NeuronMatrix): Unit = {this.data := that.data}
  def +=(that:NeuronMatrix): Unit = {this.data :+= that.data}
  def :*=(x:Double): Unit = {this.data :*= x}
  def reshape(r: Int, c: Int, isView: Boolean = true) = new NeuronMatrix(data.reshape(r,c, isView))
  def vec(isView: Boolean = true) = new NeuronVector(data.flatten(isView))  // important!
  def transpose = new NeuronMatrix(data.t)
  def set(x: Double) : Unit={data:=x; }
  def euclideanSqrNorm: Double = {val z = norm(data.flatten()); z*z}
  def sumCol() = new NeuronVector(sum(data(::,*)).toDenseVector)
  def sumRow() = new NeuronVector(sum(data(*,::)).toDenseVector)
  def sumAll():Double = sum(data)
  def colVec(i: Int) = new NeuronVector(data(::,i))
  def DOT(that: NeuronMatrix): NeuronMatrix = new NeuronMatrix(this.data :* that.data)
  def spliceRow(num: Int): (NeuronMatrix, NeuronMatrix) = (new NeuronMatrix(this.data(0 until num, ::)), new NeuronMatrix(this.data(num to -1, ::)))
  def padRow(that: NeuronMatrix) = new NeuronMatrix(DenseMatrix.vertcat(this.data, that.data))
  def Cols(range: Range) = new NeuronMatrix(data(::,range))
  def Rows(range: Range) = new NeuronMatrix(data(range, ::))
  def CROSS (that: NeuronMatrix): NeuronTensor = {
    assert(this.cols == that.cols)
    val m = new NeuronMatrix(this.rows * that.rows, this.cols)
    for (i<- 0 until that.rows) {// vercat that this.rows times
      m.Rows(i*this.rows until (i+1)*this.rows) := new NeuronMatrix(this.data(*, ::) :* that.data(i, ::).t)
    }
    new NeuronTensor(m.data, this.rows, that.rows)
  }
  def asNeuronTensor(rows:Int, cols:Int): NeuronTensor = new NeuronTensor(data, rows, cols)
  def importFromFile(filename: String): Unit = {
    val source = scala.io.Source.fromFile(filename)
    val dataBlock = source.mkString.split("\\s+").map(_.toDouble)
    source.close()
    assert(dataBlock.length == rows*cols)
    data := new DenseMatrix(rows, cols, dataBlock)
  }
}

// solution to 3-order tensor
class NeuronTensor(val data: DenseMatrix[Double], val d1: Int, val d2: Int) {
  assert(d1*d2 == data.rows)
  val d3 = data.cols
  def this(d1: Int, d2: Int, d3: Int) = this(DenseMatrix.zeros[Double](d1*d2,d3), d1, d2)
  def this(d1: Int, d2: Int, d3: Int, rand: Rand[Double]) = this(DenseMatrix.rand(d1*d2,d3,rand), d1, d2)
  def mat(isView: Boolean = true) = new NeuronMatrix({if (isView) data; else data.copy})
  def Mult(that: NeuronMatrix): NeuronMatrix = {// matrix-vector mult with batch
    assert(d2 == that.rows && d3 == that.cols)
    val m = new NeuronMatrix(d1, d3) 
    for (i<- 0 until d3) {
      m.data(::, i) := data(::,i).asDenseMatrix.reshape(d1, d2) * that.data(::,i)
    }
    m
  }
  def TransMult(that: NeuronMatrix): NeuronMatrix = {// matrix-vector mult with batch
    assert(d1 == that.rows && d3 == that.cols)
    val m = new NeuronMatrix(d1, d3) 
    for (i<- 0 until d3) {
      m.data(::, i) := data(::,i).asDenseMatrix.reshape(d1, d2).t * that.data(::,i)
    }
    m
  }
  def MultLeft(left: NeuronMatrix): NeuronTensor = {// (id)matrix-matrix mult with batch
    assert(left.cols == d1)
    new NeuronTensor((left.data * data.reshape(d1, d2*d3)).reshape(left.rows * d2, d3), left.rows, d2)
    }
  def MultRight(right: NeuronMatrix): NeuronTensor = {// matrix-(id)matrix mult with batch
    assert(right.rows == d2)
    val m = new NeuronTensor(d1, right.cols, d3)
	  for (i<- 0 until d3) {
	    m.data(::, i) := (data(::, i).asDenseMatrix.reshape(d1, d2) * right.data).flatten()
	  }
     m
    }
  def MultLeftAndRight(left: NeuronMatrix, right:NeuronMatrix) = {
    this.MultLeft(left).MultRight(right)
    }
} 


class WeightVector (data: DenseVector[Double]) extends NeuronVector(data) {
  def this(n:Int) = this(DenseVector.zeros[Double](n))
  def this(n:Int, rand: Rand[Double]) = this(DenseVector.rand(n, rand))
  def concatenate (that: WeightVector) : WeightVector = new WeightVector(DenseVector.vertcat(this.data, that.data))
  var ptr : Int = 0
  def reset(): Unit = {ptr = 0; }
  def apply(W:NeuronMatrix, b:NeuronVector): Int = {
    if (W != null) {
    val rows = W.data.rows
    val cols = W.data.cols
    
    W.data := data(ptr until ptr + rows*cols).asDenseMatrix.reshape(rows, cols)
    ptr = (ptr + rows * cols) % length
    }
    if (b != null) {
    b.data := data(ptr until ptr + b.length)
    ptr = (ptr + b.length) % length
    }
    ptr
  }
  def get(W:NeuronMatrix, b:NeuronVector): Int = {
    if (W != null) {
    val rows = W.data.rows
    val cols = W.data.cols
    
    data(ptr until ptr + rows*cols).asDenseMatrix.reshape(rows, cols) := W.data
    ptr = (ptr + rows * cols) % length
    }
    if (b != null) {
    data(ptr until ptr + b.length) := b.data
    ptr = (ptr + b.length) % length
    }
    ptr
  }
  def set(wv: NeuronVector): Int = {
    ptr = 0
    data := wv.data
    0
  }
  override def copy(): WeightVector = new WeightVector(data.copy)
}


object NullVector extends NeuronVector (0)
class OnesVector(n:Int) extends NeuronVector(DenseVector.ones[Double](n)) 

object NullMatrix extends NeuronMatrix (0,0)
class OnesMatrix(r:Int, c:Int) extends NeuronMatrix(DenseMatrix.ones[Double](r,c))



/********************************************************************************************/
// Implement batch mode training 
abstract trait Optimizable {
  /*************************************/
  // To be specified 
  var nn: InstanceOfNeuralNetwork = null
  var xData : Array[NeuronVector] = null
  var yData : Array[NeuronVector] = null
  var xDataTest : Array[NeuronVector] = null
  var yDataTest : Array[NeuronVector] = null
  
  var xDataM : NeuronMatrix = null
  var yDataM : NeuronMatrix = null
  var xDataTestM: NeuronMatrix = null
  var yDataTestM: NeuronMatrix = null
  /*************************************/
  
  final var randomGenerator = new scala.util.Random
  
  def initMemory(inn: InstanceOfNeuralNetwork = nn) : SetOfMemorables = {
    val seed = ((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString
    val mem = new SetOfMemorables
    inn.init(seed, mem).allocate(seed, mem)
    mem
  }
  
  def getRandomWeightVector (coeff: Double = 1.0, inn: InstanceOfNeuralNetwork = nn) : WeightVector = {
    
    val wdefault = inn.getRandomWeights(System.currentTimeMillis().hashCode.toString) // get dimension of weights
    val rv = new WeightVector(wdefault.length) 
    rv := (wdefault * coeff)
    rv
  }
  
  def getObj(w: WeightVector, distance:DistanceFunction = L2Distance) : Double = { // doesnot compute gradient or backpropagation
    val size = xData.length
    assert (size >= 1 && size == yData.length)
    var totalCost: Double = 0.0
    val dw = new WeightVector(w.length)
    
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)    
    (0 until size).par.foreach(i => {
      nn(xData(i),initMemory())
    })
    
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    if (yData != null) {//supervised
      totalCost = (0 until size).par.map(i => {
    	  distance(nn(xData(i), initMemory()), yData(i))
      }).reduce(_+_)
    } else {//unsupervised
      totalCost = (0 until size).par.map(i => {
          nn(xData(i), initMemory()); 0.0
      }).reduce(_+_)
    }
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)
    totalCost/size + regCost
  }
  
  def getObjAndGrad (w: WeightVector, distance:DistanceFunction = L2Distance, batchSize: Int = 0): (Double, NeuronVector) = {
    val size = xData.length
    assert(size >= 1 && (null == yData || size == yData.length))
    var totalCost:Double = 0.0
    /*
     * Compute objective and gradients in batch mode
     * which can be run in parallel 
     */
    
    val dw = new WeightVector(w.length)
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    
    var sampleArray = (0 until size).toList.par
    if (batchSize <= 0) {
      // use full-batch as default
    } else {
      sampleArray = scala.util.Random.shuffle((0 until size).toList).slice(0, batchSize).par
    }
      
    sampleArray.foreach(i => {
      nn(xData(i),initMemory())
    })
    
    
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    if (yData != null) {//supervised
      totalCost = sampleArray.map(i => {
        val mem = initMemory()
        val x = nn(xData(i), mem); val y = yData(i)
        val z = distance.grad(x, yData(i))
        nn.backpropagate(z, mem) // update dw !
        distance(x,y)
      }).reduce(_+_)
    } else {//unsupervised
      totalCost = sampleArray.map(i => {
        val mem = initMemory()
        val x = nn(xData(i), mem);
        nn.backpropagate(new NeuronVector(x.length), mem)
        0.0
        }).reduce(_+_)
    }
    /*
     * End parallel loop
     */
    
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, sampleArray.size)
    //println(totalCost/size, regCost)
    (totalCost/sampleArray.size + regCost, dw/sampleArray.size)
  }
  def getObjM(w: WeightVector, distance:DistanceFunction = L2Distance) : Double = { // doesnot compute gradient or backpropagation
    val size = xDataM.cols
    assert(size >= 1 && (null == yDataM || size == yDataM.cols))
    var totalCost:Double = 0.0

    val dw = new WeightVector(w.length)  
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)        
    nn(xDataM,initMemory())

    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    if (yDataM != null) {//supervised
    	totalCost = distance(nn(xDataM, initMemory()), yDataM)
    } else {//unsupervised
      nn(xDataM, initMemory());
      totalCost = 0.0
    }
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)
    totalCost/size + regCost
  }
  
  def getObjAndGradM (w: WeightVector, distance:DistanceFunction = L2Distance, batchSize: Int = 0): (Double, NeuronVector) = {
    val size = xDataM.cols
    assert(size >= 1 && (null == yDataM || size == yDataM.cols))
    val blockSize = 512
    val numOfBlock: Int = (size-1)/blockSize + 1
    val ranges = ((0 until (numOfBlock-1)).map(i => blockSize*i until blockSize*(i+1)) :+ (blockSize*(numOfBlock-1) until size)).par
    
    var totalCost:Double = 0.0
    
    val dw = new WeightVector(w.length)
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    ranges.map(r =>
    	nn(xDataM.Cols(r),initMemory())
    )
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    if (yDataM != null) {//supervised
      totalCost = ranges.map(r => {
        val mem = initMemory()
        val x = nn(xDataM.Cols(r), mem); val y = yDataM.Cols(r)
        val z = distance.grad(x, y)
        nn.backpropagate(z, mem) // update dw !
        distance(x,y)}).reduce(_+_)
    } else {//unsupervised
      ranges.map(r => {
        val mem = initMemory()
        val x = nn(xDataM.Cols(r), mem);
        nn.backpropagate(new NeuronMatrix(x.rows, x.cols), mem)
      })
      totalCost = 0.0
    }
    
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)
    //println(totalCost/size, regCost)
    (totalCost/size + regCost, dw/size)
  }
  
  def getApproximateObjAndGrad (w: WeightVector, distance:DistanceFunction = L2Distance) : (Double, NeuronVector) = {
    // Compute gradient using numerical approximation
    var dW = w.copy()
    for (i<- 0 until w.length) {
	  val epsilon = 0.00001
	  val w2 = w.copy
	  w2.data(i) = w.data(i) + epsilon
	  val cost1 = getObj(w2, distance)
	  w2.data(i) = w.data(i) - epsilon
	  val cost2 = getObj(w2, distance)
	  
	  dW.data(i) = (cost1 - cost2) / (2*epsilon)
	}
    (getObj(w, distance), dW)
  }
  def getApproximateObjAndGradM (w: WeightVector, distance:DistanceFunction = L2Distance) : (Double, NeuronVector) = {
    // Compute gradient using numerical approximation
    var dW = w.copy()
    for (i<- 0 until w.length) {
	  val epsilon = 0.00001
	  val w2 = w.copy
	  w2.data(i) = w.data(i) + epsilon
	  val cost1 = getObjM(w2, distance)
	  w2.data(i) = w.data(i) - epsilon
	  val cost2 = getObjM(w2, distance)
	  
	  dW.data(i) = (cost1 - cost2) / (2*epsilon)
	}
    (getObjM(w, distance), dW)
  }  
  
  object SGD {
    import breeze.math.MutableCoordinateSpace
    def apply[T](initialStepSize: Double=4, maxIter: Int=100)(implicit vs: MutableCoordinateSpace[T, Double]) :StochasticGradientDescent[T]  = {
      new SimpleSGD(initialStepSize,maxIter)
    }

    class SimpleSGD[T](eta: Double=4,
                     maxIter: Int=100)
                    (implicit vs: MutableCoordinateSpace[T, Double]) extends StochasticGradientDescent[T](eta,maxIter) {
      type History = Unit
      def initialHistory(f: StochasticDiffFunction[T],init: T)= ()
      def updateHistory(newX: T, newGrad: T, newValue: Double, f: StochasticDiffFunction[T], oldState: State) = ()
      override def determineStepSize(state: State, f: StochasticDiffFunction[T], dir: T) = {
        defaultStepSize // / math.pow(0.001*state.iter + 1, 2.0 / 3.0)
      }
    }
  }

  /*
   * Train neural network using first order minimizer (L-BFGS)
   * Please NOTE there is no regularization penalty in training 
   * Regularization usually is done in distributed modules
   */ 
  def train(w: WeightVector, maxIter:Int = 400, distance: DistanceFunction = L2Distance, batchSize: Int = 0, opt: String = "lbfgs"): (Double, WeightVector) = {

    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val (obj, grad) = getObjAndGrad(new WeightVector(x), distance, batchSize)
	    (obj, grad.data)
	  }    
    }
    
    var w2 = new WeightVector(w.length)
    if (opt == "lbfgs") {
      val lbfgs = new LBFGS[DenseVector[Double]](maxIter)
      w2 = new WeightVector(lbfgs.minimize(f, w.data))
    }
    else if (opt == "sgd") {
      val sgd =  SGD[DenseVector[Double]](1.0,maxIter)    
      w2 = new WeightVector(sgd.minimize(f, w.data))
    }
    else if (opt == "sagd") {
      val batchf = BatchDiffFunction.wrap(f)
      val sagd = new StochasticAveragedGradient[DenseVector[Double]](maxIter, 1.0)
      w2 = new WeightVector(sagd.minimize(batchf, w.data))
    }
    (f(w2.data), w2)    
  }
  def trainx(w: WeightVector, maxIter:Int = 400, distance: DistanceFunction = L2Distance, batchSize: Int = 0, opt: String = "lbfgs"): (Double, WeightVector) = {

    val f = new DiffFunction[DenseVector[Double]] {
	  def calculate(x: DenseVector[Double]) = {
	    val (obj, grad) = getObjAndGradM(new WeightVector(x), distance, batchSize)
	    (obj, grad.data)
	  }    
    }
    
    var w2 = new WeightVector(w.length)
    if (opt == "lbfgs") {
      val lbfgs = new LBFGS[DenseVector[Double]](maxIter)
      w2 = new WeightVector(lbfgs.minimize(f, w.data))
    }
    else if (opt == "sgd") {
      val sgd =  SGD[DenseVector[Double]](1.0,maxIter)    
      w2 = new WeightVector(sgd.minimize(f, w.data))
    }
    else if (opt == "sagd") {
      val batchf = BatchDiffFunction.wrap(f)
      val sagd = new StochasticAveragedGradient[DenseVector[Double]](maxIter, 1.0)
      w2 = new WeightVector(sagd.minimize(batchf, w.data))
    }
    else if (opt == "sgdm") {
      val sgdm  = new SGDmTrain()
      w2 = new WeightVector(sgdm.minimize(f, w.data))
    }
    (f(w2.data), w2)    
  }
    
  def test(w:WeightVector, distance: DistanceFunction = L2Distance): Double = {
    val size = xDataTest.length
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    val totalCost = (0 until size).par.map(i => {
      		distance(nn(xDataTest(i), initMemory()), yDataTest(i))
      	}).reduce(_+_)
    totalCost / size
  }
}