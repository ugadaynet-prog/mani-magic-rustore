package app.manimagic.rustore

// Мост между веб-приложением (app/script.js, startNativeCheckout) и RuStore Pay SDK.
// SDK инициализируется автоматически по console_app_id_value в AndroidManifest.xml —
// вручную RuStorePayClient создавать не нужно.
//
// ВАЖНО перед первой сборкой: пара точных имён (PurchaseParams, поля результата покупки)
// уточнить по официальному примеру rustore-dev/rustore-example-java-billing — документация
// RuStore на момент написания была недоступна полностью (rate limit), поэтому здесь —
// лучшее текущее понимание API. Если Kotlin-компилятор в CI укажет на несоответствие имени
// метода/поля — это ожидаемо, достаточно поправить конкретную строку.

import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import ru.rustore.sdk.pay.RuStorePayClient
import ru.rustore.sdk.pay.model.PreferredPurchaseType
import ru.rustore.sdk.pay.model.ProductId
import ru.rustore.sdk.pay.model.PurchaseParams

@CapacitorPlugin(name = "RuStoreBilling")
class RuStoreBillingPlugin : Plugin() {

    @PluginMethod
    fun purchaseProduct(call: PluginCall) {
        val productId = call.getString("productId")
        if (productId.isNullOrBlank()) {
            call.reject("productId обязателен")
            return
        }
        try {
            val params = PurchaseParams(productId = ProductId(productId))
            RuStorePayClient.instance.getPurchaseInteractor()
                .purchase(params, preferredPurchaseType = PreferredPurchaseType.ONE_STEP)
                .addOnSuccessListener { result ->
                    val ret = JSObject()
                    ret.put("purchaseId", result.purchaseId.toString())
                    ret.put("productId", productId)
                    call.resolve(ret)
                }
                .addOnFailureListener { e ->
                    call.reject(e.message ?: "purchase_failed", e)
                }
        } catch (e: Exception) {
            call.reject(e.message ?: "purchase_failed", e)
        }
    }

    // Список товаров (цены/названия) — не обязателен для оплаты как таковой (у нас цены
    // и так известны из PLANS на сервере), но полезен, если захотим показать актуальную
    // цену из консоли RuStore прямо в паивволле.
    @PluginMethod
    fun getProducts(call: PluginCall) {
        val idsArray = call.getArray("productIds")
        val ids = mutableListOf<String>()
        if (idsArray != null) {
            for (i in 0 until idsArray.length()) {
                idsArray.getString(i)?.let { ids.add(it) }
            }
        }
        RuStorePayClient.instance.getProductInteractor()
            .getProducts(ids.map { ProductId(it) })
            .addOnSuccessListener { products ->
                val ret = JSObject()
                val arr = com.getcapacitor.JSArray()
                products.forEach { p ->
                    val item = JSObject()
                    item.put("productId", p.productId.toString())
                    arr.put(item)
                }
                ret.put("products", arr)
                call.resolve(ret)
            }
            .addOnFailureListener { e -> call.reject(e.message ?: "get_products_failed", e) }
    }

    // Текущие покупки пользователя — на случай, если приложение перезапустили
    // посреди оплаты и нужно узнать её итог без повторной попытки купить.
    @PluginMethod
    fun getPurchases(call: PluginCall) {
        RuStorePayClient.instance.getPurchaseInteractor()
            .getPurchases()
            .addOnSuccessListener { purchases ->
                val ret = JSObject()
                val arr = com.getcapacitor.JSArray()
                purchases.forEach { p ->
                    val item = JSObject()
                    item.put("purchaseId", p.purchaseId.toString())
                    item.put("productId", p.productId.toString())
                    item.put("status", p.purchaseState.toString())
                    arr.put(item)
                }
                ret.put("purchases", arr)
                call.resolve(ret)
            }
            .addOnFailureListener { e -> call.reject(e.message ?: "get_purchases_failed", e) }
    }
}
