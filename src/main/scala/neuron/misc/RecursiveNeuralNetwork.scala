// Created by: Jianbo Ye, Penn State University jxy198@psu.edu
// Copyright under MIT License 2014
package neuron.misc
import scala.collection.Set
import scala.collection.SortedSet
import neuron.autoencoder._
import neuron.math._
import neuron.core._


abstract class AgglomerativeGraph {
  type DataType
  def link : (Node, Node) => (Double, Node) // need to be implemented
  
  class Node (val t:Tree, val data:DataType) {
    def this(id: Int, data:DataType) = this(new Leaf(id), data)
    var neighbors: Set [Node] = Set()
    var connectedEdges: Set [Edge] = Set() 
  }
  class Edge (val v1: Node, val v2:Node) {
    val (v, node) = link(v1,v2)
  }

  object Edge {
  	implicit val ord = new Ordering[Edge] {
    	// Required as of Scala 2.11 for reasons unknown - the companion to Ordered
    	// should already be in implicit scope
    	import scala.math.Ordered.orderingToOrdered
    	def compare(e1:Edge, e2: Edge): Int = (e1.v, e1.hashCode()) compare (e2.v, e2.hashCode())    
    }        
  }
  
  // initializing to empty
  var nodes = Set[Node]()
  var edges = SortedSet[Edge]() // 
  
  /** add a new edge between two nodes */
  def addNewEdge(n1: Node, n2: Node): Unit = {
    val e = if (n1.t.id < n2.t.id) new Edge(n1, n2) else new Edge(n2, n1)
    n1.neighbors = n1.neighbors + n2
    n2.neighbors = n2.neighbors + n1
    n1.connectedEdges = n1.connectedEdges + e
    n2.connectedEdges = n2.connectedEdges + e
  }
  
  /** replace two nodes connected to edge e by a new node */
  def mergeNodesByEdge(e: Edge): Unit = {
    val n1 = e.v1
    val n2 = e.v2
    val v = e.node // reference a merged node
    
    nodes = nodes - (n1, n2) + v
    
    
    n1.neighbors = n1.neighbors - n2
    n2.neighbors = n2.neighbors - n1
    n1.neighbors.foreach(x=>x.neighbors = x.neighbors - n1 + v) // n2.neighbors = n2.neighbors - n1 + v
    n2.neighbors.foreach(x=>x.neighbors = x.neighbors - n2 + v)
    v.neighbors = n1.neighbors ++ n2.neighbors 
    v.connectedEdges = v.neighbors.map(n=> {
      val e = if (n.t.id < v.t.id) new Edge(n, v) else new Edge(v, n)
      n.connectedEdges = n.connectedEdges + e
      e
      })
    edges = edges ++ v.connectedEdges
    edges = edges -- n1.connectedEdges -- n2.connectedEdges 
    //edges = edges.filter(x => (x.v1 != n1 && x.v1 != n2 && x.v2 != n1 && x.v2 != n2 && x.v1 != v))    
  }
  
  /** agglomerative clustering */
  def greedyMerge() : Unit = {
    while (!edges.isEmpty) {
      val e = edges.min // find the minimum proximity pair of connected nodes
      mergeNodesByEdge(e)
    }
  }
}


/** Chain data structure as agglomerative graph */
class AgglomerativeChain (f: (NeuronVector, NeuronVector) => (Double, NeuronVector)) extends AgglomerativeGraph{
  type DataType = NeuronVector
  def link = (x, y) => {
    val (v, nd) = f(x.data,y.data)
    assert(x.data.length == nd.length && y.data.length == nd.length)
    (v, new Node(new Branch(x.t,y.t), nd))
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

abstract class Tree {
  val numOfLeaves: Int
  val id: Int
  def toString() : String
}

case class Branch (val left:Tree, val right:Tree) extends Tree {
  val numOfLeaves = left.numOfLeaves + right.numOfLeaves
  val id = scala.math.min(left.id, right.id)
  override def toString() = "(" + left.toString() + " " + right.toString() + ")" 
}
case class Leaf(val id: Int = 0) extends Tree {
  val numOfLeaves: Int = 1
  override def toString() = id.toString()
}



class RecursiveNeuralNetwork (val tree: Tree, // The root node of tree
						  	  val enc: Operationable, // val enc: RecursiveClass, 
						  	  val input: Operationable) 
	extends Operationable with EncoderWorkspace {
    assert(2*input.outputDimension == enc.inputDimension)
	val inputDimension = tree.numOfLeaves * input.inputDimension
	val outputDimension = enc.outputDimension

	val (leftRNN, rightRNN) = tree match {
      case Leaf(id) => (null, null)
      case Branch(left, right) =>  (new RecursiveNeuralNetwork(left, enc, input),
          new RecursiveNeuralNetwork(right, enc, input))
    }
    
	def create() = tree match{
	  case Leaf(id) => input.create()
	  case Branch(left, right) => (enc TIMES (leftRNN PLUS rightRNN)).create()
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
class RecursiveAutoEncoder (val tree: Tree,
							val enc: InstanceOfAutoEncoder ,
							val input: InstanceOfAutoEncoder,
							val regCoeff:Double = 0.0)
		extends SelfTransform(tree.numOfLeaves*input.inputDimension) with EncoderWorkspace  {
  assert(input.encodeDimension == enc.encodeDimension)
  assert(enc.inputDimension == 2*enc.encodeDimension)
  val encodeDimension = enc.encodeDimension
  val wordLength = encodeDimension
  
  def ae(tree:Tree): RAE = tree match {
      case Leaf(id) => new RAELeaf(input)
      case Branch(left, right) => new RAEBranch(enc, ae(left), ae(right), regCoeff)
  }
  
  private val AE = ae(tree)
  def encoCreate() = AE.enco.create()
  def create() = AE.create()
}

