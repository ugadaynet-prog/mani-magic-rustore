package app.manimagic.rustore

import android.content.Intent

// Единственное место, где мы обрабатываем возврат из банковского приложения
// (SBP/SberPay) во время оплаты. Вынесено отдельно от MainActivity специально:
// если точное имя метода/интерактора у текущей версии RuStore Pay SDK отличается —
// достаточно поправить только этот файл.
//
// ПРОВЕРИТЬ при первой сборке в CI по официальному примеру
// rustore-dev/rustore-example-java-billing — Kotlin-компилятор сразу укажет
// на несоответствие, если имя устарело.
object RuStorePayClientHolder {
    fun handleNewIntent(intent: Intent) {
        ru.rustore.sdk.pay.RuStorePayClient.instance
            .getIntentInteractor()
            .proceedIntent(intent)
    }
}
