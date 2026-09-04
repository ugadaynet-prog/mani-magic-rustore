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
 * Плагин возвращает:   { mask: "<png-dataUrl grayscale 384×384>", elapsedMs: <number> }
 *
 * Модель загружается один раз лениво при первом вызове segment() и хранится
 * в статическом поле — пережива перезапуски WebView без повторной загрузки.
 *
 * Входной тензор: float32[1, 3, 384, 384], значения 0..1, порядок CHW.
 * Выходной тензор: float32[1, 1, 384, 384], сырые логиты (до сигмоиды).
 */
@CapacitorPlugin(name = "NailSegmentation")
class NailSegmentationPlugin : Plugin() {

    companion object {
        private const val MODEL_ASSET = "models/nail-unet.onnx"
        private const val INPUT_SIZE = 384

        // Пороги отсева пятен, севших мимо ногтя. Подобраны на отложенных
        // кадрах при подготовке данных, те же значения в ml/clean_masks.py.
        private const val RING_PX = 7                 // ширина кольца вокруг пятна
        private const val RING_SKIN_MIN = 0.22f       // ниже — вокруг не кожа
        private const val AREA_OUTLIER = 2.5f         // во столько раз крупнее соседей
        private const val AREA_OUTLIER_MIN_PARTS = 4  // меньше — медиана бессмысленна

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

                // 2. Масштабируем в квадрат 384×384 с вписыванием (letterbox)
                val inputBitmap = letterboxBitmap(srcBitmap, INPUT_SIZE)
                srcBitmap.recycle()

                // 3. Конвертируем пиксели → float32[1,3,384,384] CHW, 0..1
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

                // Модель возвращает 4D тензор float[1][1][384][384].
                // "Разворачиваем" его в плоский FloatArray длиной 384*384 = 65536.
                @Suppress("UNCHECKED_CAST")
                val logits4d = (results[outputName].get().value as Array<Array<Array<FloatArray>>>)
                val logits2d = logits4d[0][0]                          // float[384][384]
                val flatLogits = FloatArray(INPUT_SIZE * INPUT_SIZE)   // 65536
                for (row in 0 until INPUT_SIZE) {
                    System.arraycopy(logits2d[row], 0, flatLogits, row * INPUT_SIZE, INPUT_SIZE)
                }

                inputTensor.close()
                results.close()

                // 5. Убираем пятна, севшие мимо ногтя: вишню в руке, камень
                // в кольце, жемчужину. Тензор здесь ещё нужен — по нему
                // определяется цвет кожи вокруг каждого пятна.
                suppressStrayBlobs(flatLogits, tensor)

                // 6. Сигмоида → grayscale bitmap 384×384
                val maskBitmap = logitsToBitmap(flatLogits)

                // 7. Кодируем PNG в base64
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

    // Логиты 384×384 → grayscale Bitmap: яркость пикселя = вероятность * 255
    /**
     * Убирает из маски пятна, севшие мимо ногтя.
     *
     * Модель выучила «гладкий блестящий овал ≈ ноготь» и красит вишню в руке,
     * камень в кольце, жемчужину. Ловим это двумя правилами, теми же, что при
     * подготовке обучающих данных (ml/clean_masks.py):
     *
     *  1. Ноготь лежит на пальце, значит по его краю есть кожа. Пятно на вишне
     *     окружено вишней. Кожу считаем в YCbCr — там она занимает узкий и
     *     устойчивый диапазон по цветности, почти независимо от освещения.
     *  2. Ногти одной кисти сопоставимы по размеру, а предмет в руке заметно
     *     крупнее. Это добирает случай, когда предмет зажат в пальцах и кожа
     *     вокруг него всё-таки есть.
     *
     * На отложенных кадрах даёт IoU 0.807 -> 0.815, переобучения не требует.
     * Отвергнутым пикселям ставим большой отрицательный логит: порог в
     * интерфейсе продолжает работать для оставшихся областей.
     */
    private fun suppressStrayBlobs(logits: FloatArray, chw: FloatBuffer) {
        val n = INPUT_SIZE * INPUT_SIZE
        val plane = n
        val fg = BooleanArray(n) { sigmoid(logits[it]) > 0.5f }

        // Разметка связных областей обходом в ширину.
        val label = IntArray(n) { -1 }
        val areas = ArrayList<Int>()
        val boxes = ArrayList<IntArray>()   // x0, y0, x1, y1 на каждую область
        val queue = IntArray(n)
        for (start in 0 until n) {
            if (!fg[start] || label[start] >= 0) continue
            val id = areas.size
            var head = 0; var tail = 0
            queue[tail++] = start; label[start] = id
            var area = 0
            var x0 = INPUT_SIZE; var y0 = INPUT_SIZE; var x1 = 0; var y1 = 0
            while (head < tail) {
                val p = queue[head++]; area++
                val x = p % INPUT_SIZE; val y = p / INPUT_SIZE
                if (x < x0) x0 = x; if (x > x1) x1 = x
                if (y < y0) y0 = y; if (y > y1) y1 = y
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= INPUT_SIZE || ny >= INPUT_SIZE) continue
                    val q = ny * INPUT_SIZE + nx
                    if (fg[q] && label[q] < 0) { label[q] = id; queue[tail++] = q }
                }
            }
            areas.add(area)
            boxes.add(intArrayOf(x0, y0, x1, y1))
        }
        if (areas.isEmpty()) return

        fun isSkin(idx: Int): Boolean {
            val r = chw.get(idx) * 255f
            val g = chw.get(plane + idx) * 255f
            val b = chw.get(2 * plane + idx) * 255f
            val cb = 128f - 0.168736f * r - 0.331264f * g + 0.5f * b
            val cr = 128f + 0.5f * r - 0.418688f * g - 0.081312f * b
            return cb >= 77f && cb <= 130f && cr >= 133f && cr <= 177f
        }

        val reject = BooleanArray(areas.size)

        // Правило 1: доля кожи в кольце вокруг области.
        // Кольцо ищем только внутри рамки самой области, расширенной на RING_PX.
        // Перебор всего кадра на каждую область — это 33 миллиона проверок,
        // на телефоне заметные секунды; рамка ногтя укладывается в тысячи.
        for (id in areas.indices) {
            val b = boxes[id]
            val bx0 = maxOf(0, b[0] - RING_PX); val by0 = maxOf(0, b[1] - RING_PX)
            val bx1 = minOf(INPUT_SIZE - 1, b[2] + RING_PX)
            val by1 = minOf(INPUT_SIZE - 1, b[3] + RING_PX)
            var skin = 0; var total = 0
            for (y in by0..by1) for (x in bx0..bx1) {
                val p = y * INPUT_SIZE + x
                if (fg[p]) continue
                var touches = false
                var dy = -RING_PX
                loop@ while (dy <= RING_PX) {
                    var dx = -RING_PX
                    while (dx <= RING_PX) {
                        val nx = x + dx; val ny = y + dy
                        if (nx in 0 until INPUT_SIZE && ny in 0 until INPUT_SIZE &&
                            label[ny * INPUT_SIZE + nx] == id) { touches = true; break@loop }
                        dx++
                    }
                    dy++
                }
                if (!touches) continue
                total++
                if (isSkin(p)) skin++
            }
            if (total >= 20 && skin.toFloat() / total < RING_SKIN_MIN) reject[id] = true
        }

        // Правило 2: область заметно крупнее типичного ногтя этого кадра.
        val kept = areas.indices.filter { !reject[it] }
        if (kept.size >= AREA_OUTLIER_MIN_PARTS) {
            val sorted = kept.map { areas[it] }.sorted()
            val median = sorted[sorted.size / 2].toFloat()
            for (id in kept) if (areas[id] > AREA_OUTLIER * median) reject[id] = true
        }

        for (p in 0 until n) {
            val id = label[p]
            if (id >= 0 && reject[id]) logits[p] = -30f
        }
    }

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
