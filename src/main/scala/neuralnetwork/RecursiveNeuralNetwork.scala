package neuralnetwork

/********************************************************************************************/
// Graph data structure

abstract trait Graph {
}

abstract trait Node {
  //val neighbors: Set[Node]
}

abstract class Tree {
  val numOfLeaves: Int
}

case class Branch (val left:Tree, val right:Tree) extends Tree {
  val numOfLeaves = left.numOfLeaves + right.numOfLeaves
}
case class Leaf extends Tree with Node {
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

