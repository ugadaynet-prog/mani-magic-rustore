package app.manimagic.rustore

import android.content.Intent
import android.os.Bundle
import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(RuStoreBillingPlugin::class.java)
        super.onCreate(savedInstanceState)
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
