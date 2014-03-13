package neuralnetwork

/********************************************************************************************/
// Graph data structure

abstract trait Graph {
}

abstract trait Node {
  val neighbors: Set[Node]
}

abstract trait BinaryTreeNode {
  val left, right: BinaryTreeNode
  def isLeaf(): Boolean = false
  val numOfLeaves: Int
}
abstract trait Leaf extends BinaryTreeNode with Node {
  val left: Null = null
  val right:Null = null
  override def isLeaf(): Boolean = true
  val numOfLeaves = 1
}

class RecursiveNeuralNetwork (val tree: BinaryTreeNode, // The root node of tree
						  	  val enc: RecursiveClass, 
						  	  val input: Operationable) 
	extends Operationable with EncoderWorkspace {
    assert(input.outputDimension == enc.wordLength)
	val inputDimension = tree.numOfLeaves * input.inputDimension
	val outputDimension = enc.wordLength

	val leftRNN = if (tree.isLeaf()) null else new RecursiveNeuralNetwork(tree.left, enc, input) 
	val rightRNN= if (tree.isLeaf()) null else new RecursiveNeuralNetwork(tree.right,enc, input)
	def create() = {
	  if (tree.isLeaf()) {
	    input.create()
	  } else {
	    (enc.extract() TIMES (leftRNN PLUS rightRNN)).create()
	  }
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

