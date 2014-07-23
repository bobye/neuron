package neuron.math
import scala.collection.Set
import scala.collection.SortedSet

abstract class AgglomerativeGraph {
  type DataType
  
  /** abstract function to be implemented */
  def link : (Node, Node) => (Double, Node)
  
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
  def mergeNodesByEdge(e: Edge): (DataType, DataType) = {
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
    (n1.data, n2.data)
  }
  
  /*  agglomerative clustering: the initial edge maps could contain multiple 
      disconnected graphs such that 
      when size = 0: each connected component yields a single node after greedyMerge()
      when size > 0: it will terminate until $size data points are collected (testing) */
  def greedyMerge(size: Int = 0 ) : List[(DataType, DataType)] = {
    var dataList = List[(DataType, DataType)]()
    while (!edges.isEmpty && (dataList.length < size || size == 0)) {
      val e = edges.min // find the minimum proximity pair of connected nodes
      val twoNodes = mergeNodesByEdge(e)
      dataList =  (twoNodes._1, twoNodes._2) :: dataList
    }
    dataList
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