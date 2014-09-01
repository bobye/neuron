// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Copyright under MIT License 2014
package neuron.misc

import neuron.autoencoder._
import neuron.math._
import neuron.core._


/** Chain data structure as agglomerative graph */
class AgglomerativeChain (f: (NeuronVector, NeuronVector) => (Double, NeuronVector)) extends AgglomerativeGraph{
  type DataType = NeuronVector
  def link = (x, y) => {
    val (v, nd) = f(x.data,y.data)
    assert(x.data.length == nd.length && y.data.length == nd.length)
    (v, new Node(new BBranch(x.t,y.t), nd))
  }
  
  def loadChain(x: NeuronVector, wordLength: Int): Unit = {
    assert(x.length > 0 && x.length % wordLength == 0)
    
    var (head, xtmp) = x.splice(wordLength)
    
    var h1 = new Node(1, head)
    nodes = nodes + h1
    for (i <- 1 until x.length/wordLength) yield {      
      var (head, xtmp2) = xtmp.splice(wordLength); 
      xtmp =xtmp2
      var h2 = new Node(i+1, head)
      nodes = nodes + h2
      h1.neighbors = h1.neighbors + h2
      h2.neighbors = h2.neighbors + h1
      val e = new Edge(h1, h2)
      edges = edges + e
      h1.connectedEdges = h1.connectedEdges + e
      h2.connectedEdges = h2.connectedEdges + e
      h1 = h2
    }
  }
}

/** Optimization for ae: Recursive AE and cb: CodeBook */
class AgglomerativeAutoEncoderFactory (val ae: InstanceOfAutoEncoder, 
									   val cb: InstanceOfAutoEncoder) {
  
  var rawDataInputs: NeuronMatrix = null
  var xDataM: NeuronMatrix = null
  val aTreeGen: AgglomerativeTreeGenerator = new AgglomerativeTreeGenerator

  
  def train(): Unit = {
    rawDataInputs = aTreeGen.importFromFile("process.dat")
    
    for (i<- 0 until 100) {// run iterations
      aTreeGen.reStart()
      xDataM = aTreeGen.prepareDataForAE()
      updateNN() // update ae and in
    }
  }
  
  
  private def updateNN(): Unit = { }
  
  class AgglomerativeTreeGenerator extends AgglomerativeGraph {
    // initializing to empty
    import scala.collection.Set
    import scala.collection.SortedSet    
    var nodes_final = Set[Node]()
    var edges_final = SortedSet[Edge]() //   
    
    type DataType = NeuronVector
    def link = (x, y) => {
      val z = x.data concatenate y.data
      (L2Distance(ae(z, null), z), 
       new Node(new BBranch(x.t,y.t), ae.encode (z, null)))
    }
    

    
    def reStart(): Unit = {
      // reload data for nodes_finalset
      val size = rawDataInputs.cols
      val updatedData = cb(rawDataInputs, null)
      val it = nodes_final.iterator
      for (i<-0 until size) {
        it.next.data := updatedData.colVec(i)
      }
      // reset nodes and edges
      nodes = Set[Node]() ++ nodes_final
      edges = SortedSet[Edge]() ++ edges_final
    }
    
    def prepareDataForAE(isShuffled: Boolean = true): NeuronMatrix = {
      // generate agglomerative tree 
      val dataList = greedyMerge()
      val it = dataList.iterator
      
      val size = dataList.length
      val dim  = dataList(0)._1.length + dataList(0)._2.length
      val DataM = new NeuronMatrix(dim, size)
      val shuffedList = scala.util.Random.shuffle((0 until size).toList)
      
      for (j<- 0 until size) {
	      DataM.colVec(shuffedList(j)) := {
	        val tuple = it.next
	        tuple._1 concatenate tuple._2 // efficiency can be improved here
	      }
	    }
      DataM 
    }
    
    
    def importFromFile(filename:String): NeuronMatrix = {
      // initialize rawDataInputs, nodes_final, and edges_final
      // and return rawDataInputs
      null
    }
    
  }
  
  
}


class RecursiveNeuralNetwork (val tree: BTree, // The root node of tree
						  	  val enc: Operationable, // val enc: RecursiveClass, 
						  	  val input: Operationable) 
	extends Operationable with EncoderWorkspace {
    assert(2*input.outputDimension == enc.inputDimension)
	val inputDimension = tree.numOfLeaves * input.inputDimension
	val outputDimension = enc.outputDimension

	val (leftRNN, rightRNN) = tree match {
      case BLeaf(id) => (null, null)
      case BBranch(left, right) =>  (new RecursiveNeuralNetwork(left, enc, input),
          new RecursiveNeuralNetwork(right, enc, input))
    }
    
	def create() = tree match{
	  case BLeaf(id) => input.create()
	  case BBranch(left, right) => (enc ** (leftRNN ++ rightRNN)).create()
	}
	
	
}



abstract trait RAE extends EncodeClass with EncoderWorkspace{
  val enco: Operationable
  val deco: Operationable
}

class RAELeaf(val leaf:InstanceOfAutoEncoder) 
	extends AutoEncoder(leaf.NN.regCoeff, leaf.encoderInstance, leaf.decoderInstance) with RAE {
  val enco = leaf.extract()
  val deco = leaf.decoderInstance
}

class RAEBranch(val enc:InstanceOfAutoEncoder, 
    val leftRAE: RAE, val rightRAE: RAE, regCoeff:Double = 0.0)
	extends AutoEncoder ( 
	    regCoeff,
	    new ChainNeuralNetwork(new InstanceOfEncoderNeuralNetwork(enc), new JointNeuralNetwork(leftRAE.enco, rightRAE.enco)),
  	    new ChainNeuralNetwork(new JointNeuralNetwork(leftRAE.deco, rightRAE.deco), enc.decoderInstance)) with RAE {
  val enco = this.extract()
  val deco = decoder
}

// This is unfolding recursive autoencoder
class RecursiveAutoEncoder (val tree: BTree,
							val enc: InstanceOfAutoEncoder ,
							val input: InstanceOfAutoEncoder,
							val regCoeff:Double = 0.0)
		extends NeuralNetwork(tree.numOfLeaves*input.inputDimension) with SelfTransform with EncoderWorkspace  {
  assert(input.encodeDimension == enc.encodeDimension)
  assert(enc.inputDimension == 2*enc.encodeDimension)
  val encodeDimension = enc.encodeDimension
  val wordLength = encodeDimension
  
  def ae(tree:BTree): RAE = tree match {
      case BLeaf(id) => new RAELeaf(input)
      case BBranch(left, right) => new RAEBranch(enc, ae(left), ae(right), regCoeff)
  }
  
  private val AE = ae(tree)
  def encoCreate() = AE.enco.create()
  def create() = AE.create()
}

