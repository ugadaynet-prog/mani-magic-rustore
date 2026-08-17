package app.manimagic.rustore

import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin

/**
 * Отдаёт странице размеры системных панелей (статус-бар, панель навигации).
 *
 * Зачем это вообще нужно. На Android WebView css-функция env(safe-area-inset-*)
 * про системные панели НЕ знает — она сделана под вырез экрана на iOS и здесь
 * всегда возвращает ноль. Поэтому страница сама узнать отступы не может.
 *
 * Почему не padding на WebView, как было раньше. Так пробовали в версиях 7 и 8:
 * ставили OnApplyWindowInsetsListener и звали requestApplyInsets — сначала в
 * onCreate, потом при фактическом прикреплении view к окну. Оба раза на живом
 * телефоне контент всё равно уезжал под панели. Отлаживать вслепую дальше смысла
 * не было, поэтому схема перевёрнута: не нативная часть навязывает отступ, а
 * страница СПРАШИВАЕТ значения, когда она уже загружена и готова их применить.
 * Гонки при таком порядке не бывает по построению.
 *
 * Значения отдаём в CSS-пикселях (делим на плотность экрана), чтобы страница
 * могла подставить их в свои переменные без пересчётов.
 */
@CapacitorPlugin(name = "Insets")
class InsetsPlugin : Plugin() {

    @PluginMethod
    fun get(call: PluginCall) {
        val web = bridge?.webView
        val result = JSObject()

        if (web == null) {
            // Спросили слишком рано — отдаём нули, страница переспросит позже.
            result.put("top", 0)
            result.put("bottom", 0)
            result.put("left", 0)
            result.put("right", 0)
            result.put("ready", false)
            call.resolve(result)
            return
        }

        val bars = ViewCompat.getRootWindowInsets(web)
            ?.getInsets(WindowInsetsCompat.Type.systemBars())
        val density = web.resources.displayMetrics.density.takeIf { it > 0f } ?: 1f

        result.put("top", (bars?.top ?: 0) / density)
        result.put("bottom", (bars?.bottom ?: 0) / density)
        result.put("left", (bars?.left ?: 0) / density)
        result.put("right", (bars?.right ?: 0) / density)
        // ready=false означает «окно ещё не отдало инсеты», а не «панелей нет»:
        // без этого признака страница не отличила бы одно от другого и осталась
        // бы с нулями навсегда.
        result.put("ready", bars != null)
        call.resolve(result)
    }
}
