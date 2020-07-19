package gudu

object gudulib {
  val rand = scala.util.Random
  val sigmoid : Double => Double =  x => 1./ (1+ Math.exp(-x))
  val sigmoidPrime : Double => Double =  x => sigmoid(x)*(1-sigmoid(x))
  val fActivation = sigmoid
  val fActivationPrime = sigmoidPrime

  def MSE(v: Vector[Double], w: Vector[Double]): Double = (for ((x,y)<-v zip w) yield ((x-y)*(x-y))).sum / 2

  def getRandomMiniBatch(X_train:Vector[Vector[Double]], Y_train:Vector[Vector[Double]])={
    Vector(X_train, Y_train)
  }


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
    val size = architecture.length

    def Run(input: Vector[Double]) = architecture.foldLeft(input) { (x: Vector[Double], L: DenseLayer) => L.Compute(x, fActivation) }
    def Run

    def BackProp(x: Vector[Double], y: Vector[Double], alpha: Double =0.01 ) = {
      val yhat =  Run(x)
      val mserror =  MSE(yhat,y)

      val delta = for ( (neur, yj) <- this.architecture.last.neurons zip y) yield fActivationPrime(neur.weights)
    }
  }
}
