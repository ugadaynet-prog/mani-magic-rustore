package app.manimagic.rustore

import android.content.Intent

// Единственное место, где мы обрабатываем возврат из банковского приложения
// (SBP/SberPay) во время оплаты. Вынесено отдельно от MainActivity специально:
// если точное имя метода/интерактора у текущей версии RuStore Pay SDK отличается —
// достаточно поправить только этот файл.
//
// Имена сверены с официальной документацией (rustore.ru/help/sdk/pay/kotlin-java,
// версия 11.0.0) — getIntentInteractor().proceedIntent(intent) подтверждены.
// Используется и из onCreate (холодный старт активности после возврата из банковского
// приложения), и из onNewIntent (активность уже была жива) — по этой же документации
// нужны оба вызова, не только onNewIntent.
object RuStorePayClientHolder {
    fun handleIntent(intent: Intent) {
        ru.rustore.sdk.pay.RuStorePayClient.instance
            .getIntentInteractor()
            .proceedIntent(intent)
    }
}
