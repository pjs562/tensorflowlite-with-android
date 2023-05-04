package unist.pjs.tensorflowlite_with_android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import unist.pjs.tensorflowlite_with_android.ml.Model
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var camera: Button
    private lateinit var gallery: Button
    private lateinit var result: TextView
    private lateinit var imageView: ImageView
    private var imageSize = 32

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        camera = findViewById(R.id.button1)
        gallery = findViewById(R.id.button2)

        result  = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        camera.setOnClickListener {
            if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent,  3)
            }else{
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        gallery.setOnClickListener {
                val cameraIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                startActivityForResult(cameraIntent,  1)
        }
    }
    fun classifyImage(image: Bitmap){
        val model = Model.newInstance(applicationContext)

// Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        var intValues = IntArray(imageSize * imageSize)
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0
        //iterate over each pixel and extract R, G and B values. Add those values individually to the byte buffer.
        for(i: Int in 0 until imageSize){
            for(j: Int in 0 until imageSize){
                var value: Int = intValues[pixel++] //RGB
                byteBuffer.putFloat((value.shr(16).and(0xFF)*(1.toFloat()/1)))
                byteBuffer.putFloat((value.shr(8).and(0xFF)*(1.toFloat()/1)))
                byteBuffer.putFloat((value.and(0xFF)*(1.toFloat()/1)))
            }
        }
        //60 - 71 pixel data to byte buffer
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        var confidences = outputFeature0.floatArray
        //find the index of the class with the biggest confidence
        var maxPos = 0
        var maxConfidence: Float = 0.toFloat()
        for(i: Int in confidences.indices){
            if(confidences[i] > maxConfidence){
                maxConfidence = confidences[i]
                maxPos = i
            }
        }
        var classes = arrayOf("Apple", "Banana", "Orange")
        result.text = classes[maxPos]
        classes[maxPos]
        // Releases model resources if no longer used.
        model.close()
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                var image = data?.extras?.get("data") as Bitmap
                var dimension = image.width.coerceAtMost(image.height)
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                imageView.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                classifyImage(image)
            }else{
                var dat: Uri = data?.data as Uri
                var image: Bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, dat)
                imageView.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                classifyImage(image)
            }
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}