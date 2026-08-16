package app.manimagic.rustore

import android.content.Intent
import android.os.Bundle
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.doOnAttach
import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(RuStoreBillingPlugin::class.java)
        super.onCreate(savedInstanceState)

        // targetSdk 36 включает edge-to-edge принудительно, отказаться нельзя.
        // WebView сам по себе не сдвигает контент от статус-бара/жестовой зоны —
        // полагаться на CSS env(safe-area-inset-*) внутри страницы недостаточно
        // (проверено: не помогло даже после пересборки с актуальным CSS). Поэтому
        // применяем системные отступы как padding нативно, на самом WebView.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        bridge?.webView?.let { web ->
            // WebView заливает своим фоном всю площадь, включая зону padding —
            // без этого полосы сверху/снизу были бы белыми поверх тёмной страницы.
            web.setBackgroundColor(ContextCompat.getColor(this, R.color.appBackground))
            ViewCompat.setOnApplyWindowInsetsListener(web) { view, insets ->
                val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
            // requestApplyInsets на НЕприкреплённой view — пустышка, а в onCreate
            // WebView к окну ещё не прикреплён. Из-за этого версия 7 уехала в
            // RuStore с неработающим фиксом: слушатель стоял, но его никто ни разу
            // не дёргал, и системные панели перекрывали кнопки сверху и снизу.
            // Просим отступы в момент реального прикрепления к окну.
            web.doOnAttach { ViewCompat.requestApplyInsets(it) }
        }

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
