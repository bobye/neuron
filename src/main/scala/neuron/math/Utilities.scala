// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Last Updated: April 2014
// Copyright under MIT License
package neuron.math

import breeze.linalg._
import breeze.optimize._
import breeze.stats.distributions._
import neuron.core._
//import breeze.math._


/*******************************************************************************************/
// Implement batch mode training 
abstract trait Optimizable {
  /************************************/
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
  /************************************/
  
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
    if (yDataM != null) {//supervised
    	totalCost = distance(nn(xDataM, initMemory()), yDataM)
    } else {//unsupervised
      nn(xDataM, initMemory());
      totalCost = 0.0
    }
    
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)
    totalCost/size + regCost
  }
  
  def partitionDataRanges(size: Int, blockSize: Int) = {
    val numOfBlock: Int = (size - 1)/blockSize + 1
    (0 until (numOfBlock-1)).map(i => blockSize*i until blockSize*(i+1)) :+ (blockSize*(numOfBlock-1) until size)
  }
  
  def getObjAndGradM (w: WeightVector, distance:DistanceFunction = L2Distance, batchSize: Int = 512): (Double, NeuronVector) = {
    val size = xDataM.cols
    assert(size >= 1 && (null == yDataM || size == yDataM.cols))
    val ranges = partitionDataRanges(size, batchSize).par 
    
    var totalCost:Double = 0.0
    
    val dw = new WeightVector(w.length)
    
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
  
  
    //var currentIndex: Int = 0 
  def getObjAndGradM2L (w: WeightVector, distance:DistanceFunction = new KernelDistance(SquareFunction), batchSize: Int = 0, bufferSize: Int = 600): (Double, NeuronVector) = {
    val size = xDataM.cols
    assert(size >= 1 && (null == yDataM || size == yDataM.cols))
    val rangesAll = partitionDataRanges(size, bufferSize).par  
    

    val dw = new WeightVector(w.length)
    
    nn.setWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, w)
    
    // Shuffle! Will improve over-fitting problem
   // if (currentIndex % 100 == 0) xDataM = xDataM.shuffleCols
    
    
    val totalCost =
      rangesAll.map(r => {
        val mem = initMemory()
        val x = xDataM.Cols(r)
        val y = nn(x, mem); 
        
        val z = new NeuronMatrix(x.rows, x.cols)
        
        // Start doing batch wise matrix multiplication
        val ranges = partitionDataRanges(size, batchSize).par 
        val miniBatchCost = ranges.map(minir => {
        
          val minix = x.Cols(minir)
          val miniy = y.Cols(minir)

          val (zval,zgrad) = distance.applyWithGrad(miniy, minix)
          // compute kernel gradient and return value
          z.Cols(minir) := zgrad
          zval
        }).reduce(_+_)
        
        //BP
        nn.backpropagate(z, mem)
        
        // compute objective
        miniBatchCost
      }).reduce(_+_)
 
    val regCost = nn.getDerativeOfWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString, dw, size)

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
	import breeze.math.NormedModule
    def apply[T](initialStepSize: Double=4, maxIter: Int=100)(implicit vs: NormedModule[T, Double]) :StochasticGradientDescent[T]  = {
      new SimpleSGD(initialStepSize,maxIter)
    }

    class SimpleSGD[T](eta: Double=4,
                     maxIter: Int=100)
                    (implicit vs: NormedModule[T, Double]) extends StochasticGradientDescent[T](eta,maxIter) {
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
  def trainx(w: WeightVector, maxIter:Int = 400, distance: DistanceFunction = L2Distance, batchSize: Int = 512, opt: String = "lbfgs"): (Double, WeightVector) = {

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
      val sgdm  = new SGDmTrain(0.9, 0.01, maxIter)
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
  
  def gradCheck(tolerant: Double, distance: DistanceFunction = L2Distance): WeightVector ={
    assert(tolerant > 0)
	  val w = getRandomWeightVector()
	  
	  val (obj, grad) = getObjAndGrad(w, distance)
	  val (obj2, grad2) = getApproximateObjAndGrad(w, distance)

	  assert(scala.math.abs(obj - obj2) < tolerant && grad.checkDiff(grad2, tolerant))    
	  w
  }
  def gradCheckM(tolerant: Double, distance: DistanceFunction = L2Distance): WeightVector ={
    assert(tolerant > 0)
	  val w = getRandomWeightVector()
	  
	  val (obj, grad) = getObjAndGradM(w, distance)
	  val (obj2, grad2) = getApproximateObjAndGradM(w, distance)

	  assert(scala.math.abs(obj - obj2) < tolerant && grad.checkDiff(grad2, tolerant))    
	  w
  }  
}
