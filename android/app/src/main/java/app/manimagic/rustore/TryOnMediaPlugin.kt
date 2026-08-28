package app.manimagic.rustore

import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import android.util.Base64
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin

/** Saves the rendered try-on JPEG to the public Pictures/MANI Magic album.
 * MediaStore needs no storage permission on Android 10+ (our target devices).
 */
@CapacitorPlugin(name = "TryOnMedia")
class TryOnMediaPlugin : Plugin() {
    @PluginMethod
    fun saveImage(call: PluginCall) {
        val data = call.getString("data")
        if (data.isNullOrBlank()) {
            call.reject("Не переданы данные изображения")
            return
        }
        val requestedName = call.getString("name") ?: "MANI-Magic-${System.currentTimeMillis()}.jpg"
        val fileName = if (requestedName.lowercase().endsWith(".jpg")) requestedName else "$requestedName.jpg"

        try {
            val bytes = Base64.decode(data.substringAfter(',', data), Base64.DEFAULT)
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MANI Magic")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }
            val resolver = context.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("Android не создал файл в галерее")
            try {
                resolver.openOutputStream(uri)?.use { it.write(bytes) }
                    ?: throw IllegalStateException("Не удалось открыть файл для записи")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                }
                call.resolve(JSObject().apply { put("uri", uri.toString()); put("name", fileName) })
            } catch (e: Exception) {
                resolver.delete(uri, null, null)
                throw e
            }
        } catch (e: Exception) {
            call.reject("Не удалось сохранить изображение", e)
        }
    }
}
