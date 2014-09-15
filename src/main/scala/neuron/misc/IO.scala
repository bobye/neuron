package neuron.misc

import neuron.math._
import java.awt.image._
import java.io._
import javax.imageio.ImageIO

object IO {
  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def scale_to_unit_interval(ndar: NeuronVector, eps: Double =1e-8): NeuronVector = {
    val ndar2 = ndar.copy()
    ndar2 :-= ndar2.min()
    ndar2 :*= 1.0 / (ndar2.max() + eps)
    ndar2
  }
  def tile_raster_images(X: NeuronMatrix, 
		  				 image_shape: (Int, Int),
		  				 tile_shape: (Int, Int), 
		  				 tile_spacing: (Int, Int) = (0,0),
		  				 transpose: Boolean = false): NeuronMatrix = {
    assert(image_shape._1 * image_shape._2 == X.rows)
    assert(tile_shape._1 * tile_shape._2 <= X.cols)
    val out_shape = ((image_shape._1 + tile_spacing._1) * tile_shape._1 - tile_spacing._1,
                     (image_shape._2 + tile_spacing._2) * tile_shape._2 - tile_spacing._2)
    val H = image_shape._1; val W = image_shape._2
    val Hs = tile_spacing._1; val Ws = tile_spacing._2
    
    val Y = new NeuronMatrix (out_shape._1, out_shape._2)
    for (tile_row <- 0 until tile_shape._1) {
      for (tile_col <- 0 until tile_shape._2) 
        if (tile_row * tile_shape._2 + tile_col < X.cols){
          val this_x = X.colVec(tile_row * tile_shape._2 + tile_col)
          val this_image = scale_to_unit_interval(this_x).asNeuronMatrix(image_shape._1, image_shape._2)
          val this_image_o =
          if (transpose) {
            this_image.transpose
          } else {
            this_image
          }
          Y(tile_row * (H + Hs) until tile_row * (H + Hs) + H, 
            tile_col * (W + Ws) until tile_col * (W + Ws) + W) = this_image_o
        }
      }        
    Y
  }
  
  def writeImage(filename: String)(X: NeuronMatrix): Unit = {
    import java.awt.image._
    import javax.imageio.ImageIO
    val values = X.data.data.map(d => (d * 255).toInt)
    val img: BufferedImage = new BufferedImage(X.rows, X.cols, BufferedImage.TYPE_BYTE_GRAY)
    val raster = img.getRaster()
    raster.setSamples(0, 0, X.rows, X.cols, 0, values)
    ImageIO.write(img, "PNG", new File(filename))
  }
}