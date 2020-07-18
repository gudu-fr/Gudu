object DenseNN {
  val rand = scala.util.Random
  val sigmoid : Double => Double =  x => 1./ (1+ Math.exp(-x))
  val sigmoidPrime : Double => Double =  x => sigmoid(x)*(1-sigmoid(x))
  val fActivation = sigmoid
  val fActivationPrime = sigmoidPrime

  def Normalize(v: Vector[Double]):Vector[Double] = {
    val mean = v.sum / v.size
    val std = math.sqrt(v.map( x => (x - mean)*(x - mean)).sum / v.size)
    v.map( x => (x - mean) / std)
  }

  class Neuron(nbLinked: Int) {
    var weights: Vector[Double] = Normalize((for (i <- 1 to nbLinked) yield rand.nextDouble()).to(Vector))
    var bias: Double = -1 + 2*rand.nextDouble()
    def Compute(input : Vector[Double], Activation: Double => Double):Double = {
      Activation((for ((x,y)<-Normalize(input) zip weights) yield x*y).sum  + bias)
    }
  }

  class DenseLayer(nbNeurons: Int, nbPrevNeurons: Int) {
    val size = nbNeurons
    var neurons : Vector[Neuron] = (for (i <- 1 to nbNeurons) yield new Neuron(nbPrevNeurons)).to(Vector)

    def Compute(input: Vector[Double], Activation: Double => Double): Vector[Double] = {
      for (neuron <- neurons) yield neuron.Compute(input, Activation)
    }

  }

  class DenseNetwork(arch : Vector[Int]) {

    val architecture: Vector[DenseLayer] = for ((n, m) <- (arch.drop(1) zip arch.dropRight(1))) yield new DenseLayer(n, m)

    def Run(input: Vector[Double]) = architecture.foldLeft(input) { (x: Vector[Double], L: DenseLayer) => L.Compute(x, fActivation) }
  }

  def main(arg: Array[String]): Unit ={

    /*Definition of the network*/


    //Input:
    var Input: Vector[Double] = Vector(0.0,0.5,-0.5, 0.12)
    var Input2: Vector[Double] = Vector(0.0,0.5,-0.5, 0.1)
    var Input3: Vector[Double] = Vector(0.0,-10,0.5, 0.2)

    println("Computation test")
    val DNN = new DenseNetwork(Vector(Input.size, 8, 16, 32, 16, 1))
    println(DNN.Run(Input))
    println(DNN.Run(Input2))
    println(DNN.Run(Input3))
  }
}
