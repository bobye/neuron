package neuron.math

/********************************************************************************************/
// Numerical classes and their operations : interface to breeze
import breeze.generic._
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats.distributions._
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


