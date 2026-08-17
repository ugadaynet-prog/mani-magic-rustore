package app.manimagic.rustore

import android.content.Intent
import android.os.Bundle
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(RuStoreBillingPlugin::class.java)
        registerPlugin(InsetsPlugin::class.java)
        super.onCreate(savedInstanceState)

        // targetSdk 36 включает edge-to-edge принудительно, отказаться нельзя:
        // окно всегда во весь экран, под системными панелями.
        //
        // Отступы под панели теперь расставляет САМА СТРАНИЦА — она спрашивает их
        // размеры через InsetsPlugin и подставляет в свои CSS-переменные. Версии 7
        // и 8 пытались решать это нативно, padding'ом на WebView: сначала с
        // requestApplyInsets в onCreate, потом при прикреплении view к окну. Оба
        // раза на живом телефоне контент всё равно оставался под панелями, поэтому
        // от нативного padding отказались совсем — иначе отступы сложились бы с
        // теми, что ставит страница.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        bridge?.webView?.setBackgroundColor(
            ContextCompat.getColor(this, R.color.appBackground)
        )

        // Холодный старт по возврату из банковского приложения (SBP/SberPay) —
        // отдельно от onNewIntent, иначе теряется, если систем убила активность.
        if (savedInstanceState == null) {
            try {
                RuStorePayClientHolder.handleIntent(intent)
            } catch (e: Exception) {
                // Не критично для покупок картой — не роняем активность из-за этого.
            }
        }
    }

    // Возврат из банковского приложения (SBP/SberPay) в момент оплаты, когда
    // активность уже была жива — RuStore рекомендует обрабатывать это всегда.
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        try {
            RuStorePayClientHolder.handleIntent(intent)
        } catch (e: Exception) {
            // Не критично для покупок картой — не роняем активность из-за этого.
        }
    }
}
