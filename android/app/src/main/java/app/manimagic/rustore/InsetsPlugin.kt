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

        val root = ViewCompat.getRootWindowInsets(web)
        val density = web.resources.displayMetrics.density.takeIf { it > 0f } ?: 1f

        // Берём максимум по нескольким типам, а не только systemBars.
        // Версия 1.3.1 на живом устройстве получила верный НИЖНИЙ отступ и
        // нулевой верхний — значит на этом телефоне статус-бар по systemBars()
        // не отдался. Что именно его отдаёт, вслепую не выяснить, поэтому
        // спрашиваем все подходящие типы и берём наибольшее значение: лишнего
        // это не добавит (типы описывают одни и те же панели), а пропустить
        // панель уже не даст.
        val types = intArrayOf(
            WindowInsetsCompat.Type.systemBars(),
            WindowInsetsCompat.Type.statusBars(),
            WindowInsetsCompat.Type.navigationBars(),
            WindowInsetsCompat.Type.displayCutout(),
        )
        var top = 0; var bottom = 0; var left = 0; var right = 0
        for (t in types) {
            val i = root?.getInsets(t) ?: continue
            if (i.top > top) top = i.top
            if (i.bottom > bottom) bottom = i.bottom
            if (i.left > left) left = i.left
            if (i.right > right) right = i.right
        }

        result.put("top", top / density)
        result.put("bottom", bottom / density)
        result.put("left", left / density)
        result.put("right", right / density)
        // Отдельно отдаём сырые значения по systemBars — чтобы на устройстве
        // было видно, отличаются ли они от максимума, и какой тип сработал.
        val sys = root?.getInsets(WindowInsetsCompat.Type.systemBars())
        result.put("sysTop", (sys?.top ?: 0) / density)
        result.put("sysBottom", (sys?.bottom ?: 0) / density)
        // ready=false означает «окно ещё не отдало инсеты», а не «панелей нет»:
        // без этого признака страница не отличила бы одно от другого и осталась
        // бы с нулями навсегда.
        result.put("ready", root != null)
        call.resolve(result)
    }
}
