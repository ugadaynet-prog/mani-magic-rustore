package app.manimagic.rustore

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer

/**
 * Нативный плагин сегментации ногтей.
 *
 * JavaScript вызывает: Capacitor.Plugins.NailSegmentation.segment({ image: "<jpeg-dataUrl>" })
 * Плагин возвращает:   { mask: "<png-dataUrl grayscale 256×256>", elapsedMs: <number> }
 *
 * Модель загружается один раз лениво при первом вызове segment() и хранится
 * в статическом поле — пережива перезапуски WebView без повторной загрузки.
 *
 * Входной тензор: float32[1, 3, 256, 256], значения 0..1, порядок CHW.
 * Выходной тензор: float32[1, 1, 256, 256], сырые логиты (до сигмоиды).
 */
@CapacitorPlugin(name = "NailSegmentation")
class NailSegmentationPlugin : Plugin() {

    companion object {
        private const val MODEL_ASSET = "models/nail-unet.onnx"
        private const val INPUT_SIZE = 256

        @Volatile private var ortEnv: OrtEnvironment? = null
        @Volatile private var ortSession: OrtSession? = null
        private val sessionLock = Any()

        private fun sigmoid(x: Float): Float = 1f / (1f + kotlin.math.exp(-x))
    }

    // Загружает сессию один раз; потокобезопасно через double-checked locking.
    private fun getSession(): OrtSession {
        if (ortSession != null) return ortSession!!
        synchronized(sessionLock) {
            if (ortSession != null) return ortSession!!
            val env = OrtEnvironment.getEnvironment()
            ortEnv = env
            val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }
            ortSession = env.createSession(bytes, opts)
            return ortSession!!
        }
    }

    @PluginMethod
    fun segment(call: PluginCall) {
        val dataUrl = call.getString("image")
        if (dataUrl.isNullOrBlank()) {
            call.reject("Параметр image обязателен")
            return
        }

        // Запускаем в фоновом потоке — ONNX блокирующий.
        Thread {
            try {
                val started = System.currentTimeMillis()

                // 1. Декодируем JPEG из dataURL
                val b64 = dataUrl.substringAfter(',', dataUrl)
                val jpegBytes = Base64.decode(b64, Base64.DEFAULT)
                val srcBitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                    ?: throw IllegalArgumentException("Не удалось декодировать изображение")

                // 2. Масштабируем в квадрат 256×256 с вписыванием (letterbox)
                val inputBitmap = letterboxBitmap(srcBitmap, INPUT_SIZE)
                srcBitmap.recycle()

                // 3. Конвертируем пиксели → float32[1,3,256,256] CHW, 0..1
                val tensor = bitmapToTensor(inputBitmap)
                inputBitmap.recycle()

                // 4. Запускаем инференс
                val session = getSession()
                val inputName = session.inputNames.iterator().next()
                val env = ortEnv ?: OrtEnvironment.getEnvironment()
                val inputTensor = OnnxTensor.createTensor(env, tensor,
                    longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

                val results = session.run(mapOf(inputName to inputTensor))
                val outputName = session.outputNames.iterator().next()

                // Модель возвращает 4D тензор float[1][1][256][256].
                // Приводим к Array<Array<Array<FloatArray>>> и берём logits[0][0][0] → FloatArray(65536).
                @Suppress("UNCHECKED_CAST")
                val logits4d = (results[outputName].get().value as Array<Array<Array<FloatArray>>>)

                inputTensor.close()
                results.close()

                // 5. Сигмоида → grayscale bitmap 256×256
                val maskBitmap = logitsToBitmap(logits4d[0][0][0])

                // 6. Кодируем PNG в base64
                val pngOut = ByteArrayOutputStream()
                maskBitmap.compress(Bitmap.CompressFormat.PNG, 100, pngOut)
                maskBitmap.recycle()
                val maskB64 = Base64.encodeToString(pngOut.toByteArray(), Base64.NO_WRAP)
                val elapsedMs = System.currentTimeMillis() - started

                val ret = JSObject()
                ret.put("mask", "data:image/png;base64,$maskB64")
                ret.put("elapsedMs", elapsedMs)
                call.resolve(ret)

            } catch (e: Exception) {
                call.reject("Ошибка сегментации: ${e.message}", e)
            }
        }.start()
    }

    // Вписывает bitmap в квадрат side×side с чёрными полями — тот же letterbox,
    // что и в tryon.js на стороне JS (препроцессинг идентичен тренировочному).
    private fun letterboxBitmap(src: Bitmap, side: Int): Bitmap {
        val scale = side.toFloat() / maxOf(src.width, src.height)
        val dw = (src.width * scale).toInt()
        val dh = (src.height * scale).toInt()
        val ox = (side - dw) / 2
        val oy = (side - dh) / 2

        val out = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        canvas.drawColor(android.graphics.Color.BLACK)
        val dst = android.graphics.RectF(ox.toFloat(), oy.toFloat(),
            (ox + dw).toFloat(), (oy + dh).toFloat())
        canvas.drawBitmap(src, null, dst, null)
        return out
    }

    // Bitmap ARGB → FloatBuffer CHW [R…R, G…G, B…B], значения 0..1
    private fun bitmapToTensor(bmp: Bitmap): FloatBuffer {
        val n = INPUT_SIZE * INPUT_SIZE
        val pixels = IntArray(n)
        bmp.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val buf = FloatBuffer.allocate(3 * n)
        for (i in 0 until n) buf.put(i,         ((pixels[i] shr 16) and 0xFF) / 255f) // R
        for (i in 0 until n) buf.put(n + i,      ((pixels[i] shr 8)  and 0xFF) / 255f) // G
        for (i in 0 until n) buf.put(2 * n + i,  ( pixels[i]         and 0xFF) / 255f) // B
        return buf
    }

    // Логиты 256×256 → grayscale Bitmap: яркость пикселя = вероятность * 255
    private fun logitsToBitmap(logits: FloatArray): Bitmap {
        val bmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in pixels.indices) {
            val v = (sigmoid(logits[i]) * 255f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        bmp.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        return bmp
    }
}
