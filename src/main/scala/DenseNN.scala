object DenseNN {
  import gudu.gudulib._

  def main(arg: Array[String]): Unit ={

    /*Definition of the network*/


    //Input:
    var Input: Vector[Double] = Vector(10.0,0.5,-0.50, 0.12)
    var Input2: Vector[Double] = Vector(1.0,0.5,-0.5, 0.1)
    var Input3: Vector[Double] = Vector(0.0,-10,0.5, 0.200)

    println("Computation test")
    var DNN = new DenseNetwork(Vector(Input.size, 8, 16, 32, 16, 6))
    println(DNN.Prop(Input))
    println(DNN.Run(Input))
    println(DNN.Run(Input2))
    println(DNN.Run(Input3))
    DNN.BackProp(Input, DNN.Run(Input3))
  }
}
