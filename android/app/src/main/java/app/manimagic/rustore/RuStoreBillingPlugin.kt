package app.manimagic.rustore

// Мост между веб-приложением (app/script.js, startNativeCheckout) и RuStore Pay SDK.
// SDK инициализируется автоматически по console_app_id_value в AndroidManifest.xml —
// вручную RuStorePayClient создавать не нужно.
//
// Имена сверены с официальной документацией (rustore.ru/help/sdk/pay/kotlin-java, версия
// 11.0.0): класс параметров покупки — ProductPurchaseParams (не PurchaseParams), у результата
// покупки поле purchaseId; addOnFailureListener отдаёт Throwable (не Exception) — Capacitor
// PluginCall.reject() такой overload не понимает, поэтому передаём только e.message. Purchase
// (результат getPurchases) — интерфейс с полем status; productId есть только у подтипа
// ProductPurchase (наши товары все такие — не подписки RuStore, а разовые покупки).

import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import ru.rustore.sdk.pay.RuStorePayClient
import ru.rustore.sdk.pay.model.PreferredPurchaseType
import ru.rustore.sdk.pay.model.ProductId
import ru.rustore.sdk.pay.model.ProductPurchase
import ru.rustore.sdk.pay.model.ProductPurchaseParams

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
            val params = ProductPurchaseParams(productId = ProductId(productId))
            RuStorePayClient.instance.getPurchaseInteractor()
                .purchase(params, preferredPurchaseType = PreferredPurchaseType.ONE_STEP)
                .addOnSuccessListener { result ->
                    val ret = JSObject()
                    ret.put("purchaseId", result.purchaseId.toString())
                    ret.put("productId", productId)
                    call.resolve(ret)
                }
                .addOnFailureListener { e ->
                    call.reject(e.message ?: "purchase_failed")
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
            .addOnFailureListener { e -> call.reject(e.message ?: "get_products_failed") }
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
                    // Наши товары — все разовые покупки (ProductPurchase), не подписки RuStore;
                    // productId есть только на этом подтипе, не на базовом интерфейсе Purchase.
                    item.put("productId", (p as? ProductPurchase)?.productId?.toString() ?: "")
                    item.put("status", p.status.toString())
                    arr.put(item)
                }
                ret.put("purchases", arr)
                call.resolve(ret)
            }
            .addOnFailureListener { e -> call.reject(e.message ?: "get_purchases_failed") }
    }
}
