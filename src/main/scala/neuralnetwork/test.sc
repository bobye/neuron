package neuralnetwork

object test extends Workspace {
	def inputDimension = 10                   //> inputDimension: => Int
	def outputDimension = 10                  //> outputDimension: => Int
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet
  var a = new SingleLayerNeuralNetwork(IndentityFunction, 10)
                                                  //> a  : neuralnetwork.SingleLayerNeuralNetwork = ?135085423
  var b = new LinearNeuralNetwork(10,10)          //> b  : neuralnetwork.LinearNeuralNetwork = ?1223566071
 
  var c = (a TIMES b).create()                    //> c  : neuralnetwork.InstanceOfChainNeuralNetwork[neuralnetwork.SingleLayerNeu
                                                  //| ralNetwork,neuralnetwork.LinearNeuralNetwork] = #771153740 * (#717098535)
  var d = (b PLUS c) TIMES a                      //> d  : neuralnetwork.ChainNeuralNetwork[neuralnetwork.JointNeuralNetwork[neura
                                                  //| lnetwork.LinearNeuralNetwork,neuralnetwork.InstanceOfChainNeuralNetwork[neur
                                                  //| alnetwork.SingleLayerNeuralNetwork,neuralnetwork.LinearNeuralNetwork]],neura
                                                  //| lnetwork.SingleLayerNeuralNetwork] = (?1223566071 + #771153740 * (#717098535
                                                  //| )) * (?135085423)
  
  var e = (d PLUS d);                             //> e  : neuralnetwork.JointNeuralNetwork[neuralnetwork.ChainNeuralNetwork[neura
                                                  //| lnetwork.JointNeuralNetwork[neuralnetwork.LinearNeuralNetwork,neuralnetwork.
                                                  //| InstanceOfChainNeuralNetwork[neuralnetwork.SingleLayerNeuralNetwork,neuralne
                                                  //| twork.LinearNeuralNetwork]],neuralnetwork.SingleLayerNeuralNetwork],neuralne
                                                  //| twork.ChainNeuralNetwork[neuralnetwork.JointNeuralNetwork[neuralnetwork.Line
                                                  //| arNeuralNetwork,neuralnetwork.InstanceOfChainNeuralNetwork[neuralnetwork.Sin
                                                  //| gleLayerNeuralNetwork,neuralnetwork.LinearNeuralNetwork]],neuralnetwork.Sing
                                                  //| leLayerNeuralNetwork]] = ((?1223566071 + #771153740 * (#717098535)) * (?1350
                                                  //| 85423) + (?1223566071 + #771153740 * (#717098535)) * (?135085423))
  println(e);                                     //> ((?1223566071 + #771153740 * (#717098535)) * (?135085423) + (?1223566071 + #
                                                  //| 771153740 * (#717098535)) * (?135085423))
  println(e.create())                             //> #1254691612 + #771153740 * (#717098535) * (#1459992991) + #954049115 + #7711
                                                  //| 53740 * (#717098535) * (#1590567303)
                                     
  
                                  
}