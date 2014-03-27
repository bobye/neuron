package neuralnetwork

/********************************************************************************************/
// Graph data structure
import scala.collection.SortedSet
import scala.collection.Set

abstract class GreedyMergeGraph {
  type DataType
  def link : (Node, Node) => (Double, Node) // need to be implemented
  
  class Node (val t:Tree, val data:DataType) {
    var neighbors: Set [Node] = Set()
  }
  class Edge (val v1: Node, val v2:Node) {
    val (v, node) = link(v1,v2)
  }
  
  // initializing to empty
  var nodes = Set[Node]()
  var edges = SortedSet[Edge]()(Ordering[Double].on[Edge](_.v))
  
  def greedyMerge() : Unit = {
    while (!edges.isEmpty) {
    
    val e = edges.min(Ordering.by((p:Edge)=>p.v)) // find the minimum proximity pair of connected nodes
    val n1 = e.v1
    val n2 = e.v2
    val v = e.node // reference a merged node
    
    nodes = nodes - (n1, n2) + v
    v.neighbors = n1.neighbors ++ n2.neighbors - (n1, n2) 
    n1.neighbors.foreach(x=>x.neighbors = x.neighbors - n1 + v)
    n2.neighbors.foreach(x=>x.neighbors = x.neighbors - n2 + v)
    
    edges = edges ++ n1.neighbors.map(new Edge(_, v)) ++ n2.neighbors.map(new Edge(_, v))
    edges = edges.filter(x => (x.v1 != n1 && x.v1 != n2 && x.v2 != n1 && x.v2 != n2 && x.v1 != v))
    }
    
  }
}

class GreedyMergeChain (f: (NeuronVector, NeuronVector) => (Double, NeuronVector)) extends GreedyMergeGraph{
  type DataType = NeuronVector
  def link = (x, y) => {
    val (v, nd) = f(x.data,y.data)
    assert(x.data.length == nd.length && y.data.length == nd.length)
    (v, new Node(new Branch(x.t,y.t), nd))
  }
  
  def loadChain(x: NeuronVector, wordLength: Int): Unit = {
    assert(x.length > 0 && x.length % wordLength == 0)
    
    
    var (head, xtmp) = x.splice(wordLength)
    
    var h1 = new Node(new Leaf, head)
    nodes = nodes + h1
    for (i <- 1 until x.length/wordLength ) {
      val (head, xtmp2) = xtmp.splice(wordLength); xtmp =xtmp2
      val h2 = new Node(new Leaf, head)
      nodes = nodes + h2
      h1.neighbors = h1.neighbors + h2
      h2.neighbors = h2.neighbors + h1
      edges = edges + new Edge(h1, h2)
      h1 = h2
    }
  }
}

abstract class Tree {
  val numOfLeaves: Int
}

case class Branch (val left:Tree, val right:Tree) extends Tree {
  val numOfLeaves = left.numOfLeaves + right.numOfLeaves
}
case class Leaf extends Tree {
  val numOfLeaves: Int = 1
}



class RecursiveNeuralNetwork (val tree: Tree, // The root node of tree
						  	  val enc: RecursiveClass, 
						  	  val input: Operationable) 
	extends Operationable with EncoderWorkspace {
    assert(input.outputDimension == enc.wordLength)
	val inputDimension = tree.numOfLeaves * input.inputDimension
	val outputDimension = enc.wordLength

	val (leftRNN, rightRNN) = tree match {
      case Leaf() => (null, null)
      case Branch(left, right) =>  (new RecursiveNeuralNetwork(left, enc, input),
          new RecursiveNeuralNetwork(right, enc, input))
    }
    
	def create() = tree match{
	  case Leaf() => input.create()
	  case Branch(left, right) => (enc.extract() TIMES (leftRNN PLUS rightRNN)).create()
	}
	
	
}

/*
class ContextAwareRNN (val tree:BinaryTreeNode,
					   val enc:CREncodeClass,
					   val input:Operationable,
					   val inputContext:Operationable)
    extends Operationable {
  assert(input.outputDimension == enc.wordLength)
}
* 
*/

