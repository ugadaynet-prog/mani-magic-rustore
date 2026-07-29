package app.manimagic.rustore

import android.content.Intent
import android.os.Bundle
import com.getcapacitor.BridgeActivity

class MainActivity : BridgeActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(RuStoreBillingPlugin::class.java)
        super.onCreate(savedInstanceState)
    }

    // Возврат из банковского приложения (SBP/SberPay) в момент оплаты — актуально
    // не для всех способов оплаты, но RuStore рекомендует обрабатывать это всегда.
    // ПРОВЕРИТЬ при первой сборке: точное имя интерактора/метода в текущей версии SDK.
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        try {
            RuStorePayClientHolder.handleNewIntent(intent)
        } catch (e: Exception) {
            // Не критично для покупок картой — не роняем активность из-за этого.
        }
    }
}
