package neuron.examples

import neuron.core._
import neuron.math._
import breeze.stats.distributions._
import breeze.linalg.reshape

object MLP_MNIST extends Workspace with Optimizable{

  var xDataM : NeuronMatrix = null
  var yDataM : NeuronMatrix = null
  
  private def RandomWalkDynamics(x0: NeuronMatrix, y0: NeuronMatrix, alpha: Double, copies: Int): 
	  (NeuronMatrix, NeuronMatrix) = {
    val z = new NeuronMatrix(x0.rows * x0.cols, copies, new Gaussian(0, alpha))
    val w = new NeuronMatrix(y0.rows * y0.cols, copies)
    ((z AddWith x0.vec(false)).reshape(x0.rows, x0.cols * copies),
     (w AddWith y0.vec(false)).reshape(y0.rows, y0.cols * copies))
  }	
  private def HamiltonianDynamics(x0: NeuronMatrix, 
		  						  y0: NeuronMatrix,
		  						  deltaT: Double,
		  						  ticks: Int,
		  						  alpha: Double,
		  						  copies: Int): (NeuronMatrix, NeuronMatrix) = {
    val x = x0.copy();
    val sigma2 = alpha*alpha
    val x_out = new NeuronMatrix(x0.rows, x0.cols * copies)
    
    
    
    val mem = initMemory();
    val f: NeuronMatrix => NeuronMatrix = {xx =>      
      val (cost1, zz1) = SoftMaxDistance.applyWithGradV(nn(xx, mem), y0);
      val (cost2, zz2) = L2Distance.applyWithGradV(xx, x0)
      //val costex = ExpFunction(cost1)

      val grad1 = nn.backpropagate(zz1 MultElemTransWith cost2, mem)
      val grad2 = (zz2 MultElemTransWith cost1)
      //println(grad1.euclideanSqrNormCol)
      //println(grad2.euclideanSqrNormCol)
      grad1 += grad2
    }
    
    
    val beta: Double  = 10.0
    val cost0 = new NeuronVector(x0.cols)
    for (iter <- 0 until copies) {
      val z = new NeuronMatrix(x0.rows, x0.cols, new Gaussian(0, alpha))
      cost0 += (z.euclideanSqrNormCol :/= (2*sigma2)) 
      val xs = x.copy()
      // leapfrog iteration      
      z -= f(x) * (deltaT * beta/2)
      for (i <- 0 until ticks) {      
        x += z * (deltaT / sigma2)
        z -= f(x) * (deltaT * beta)     
      }
      x += z * (deltaT / (2*sigma2))
    
      val costN1 = (z.euclideanSqrNormCol :/= (2*sigma2))
      val costN2 = SoftMaxDistance.applyV(nn(x, mem), y0) *= L2Distance.applyV(x, x0) :*= beta
      //println(cost0(0 until 10).data)
      //println(costN1(0 until 10).data)
      //println(costN2(0 until 10).data)
      val u0 = ExpFunction(cost0 -= costN1 -= costN2)
      println(u0(0 until 10).data)
      
      xs := x select (xs, u0 -= new NeuronVector(x0.cols, new Uniform(0,1)))
    
      cost0 := SoftMaxDistance.applyV(nn(xs, mem), y0) *= L2Distance.applyV(xs, x0) :*= beta
      
      x_out.Cols(iter*x0.cols until (iter+1)* x0.cols) := xs
    }
    
    val w = new NeuronMatrix(y0.rows * y0.cols, copies)
    (x_out, (w AddWith y0.vec(false)).reshape(y0.rows, y0.cols * copies))
  }

  def process(batchSize: Int = 0, setType: String, setMeta: String): WeightVector = {    
    val theta0 = nn.getRandomWeights("get random weights").toWeightVector()
    val theta_pre = theta0.copy();
    theta_pre.importFromFile("weights-pre.txt")
    nn.setWeights("set pre weight", theta_pre);
    // Load Dataset
    val data = LoadData.mnistDataM(setType, setMeta)
    
    xDataM = data._1.Cols(0 until 1000)
    yDataM = data._2.Cols(0 until 1000)
    
    import neuron.misc.IO
    IO.writeImage("tmp/samples")(
        IO.tile_raster_images(xDataM, 
                              (28, 28),
                              (20, 20),
                              (2, 2), false))     
    //val co = RandomWalkDynamics(xDataM, yDataM, 0.5, 10) 
    val co = HamiltonianDynamics(xDataM, yDataM, 0.002, 100, 0.4, 10)
    xDataM = co._1
    yDataM = co._2
    import neuron.misc.IO
    IO.writeImage("tmp/corrupted_samples")(
        IO.tile_raster_images(xDataM.Cols(9000 until 10000), 
                              (28, 28),
                              (20, 20),
                              (2, 2), false))    
    
     
    
    val (_, theta) = trainx(xDataM, yDataM, theta0, 500, SoftMaxDistance)
    println((yDataM.argmaxCol().data :== nn(xDataM, null).argmaxCol().data).activeSize / xDataM.cols.toDouble)
    xDataM = null
    yDataM = null
    theta
  }
  
  val L1 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(784,10).create()
  //val L2 = new SingleLayerNeuralNetwork(10) ** new LinearNeuralNetwork(100,10).create()  
  def main(args: Array[String]): Unit = {
    // Create NN structure

    nn = L1.create() // (L2 ** L1).create()
    
    process(500, args(0), args(1))
        
    val dataTest = LoadData.mnistDataM(args(2), args(3))
    val xDataTestM = dataTest._1
    val yDataTestM = dataTest._2
    /* compute test accuracy */
    println((yDataTestM.argmaxCol().data :== nn(xDataTestM, null).argmaxCol().data).activeSize / xDataTestM.cols.toDouble)
    
    /* export learned weights */
    import java.io._
    import neuron.misc.IO
    IO.printToFile(new File("weights.txt"))(p => 
          p.println(nn.getWeights(((randomGenerator.nextInt()*System.currentTimeMillis())%100000).toString))) 
  }

}