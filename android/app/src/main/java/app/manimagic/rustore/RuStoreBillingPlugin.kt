package app.manimagic.rustore

import com.getcapacitor.Plugin
import com.getcapacitor.annotation.CapacitorPlugin

/**
 * STUB: RuStore Pay SDK temporarily disabled (maven repo 404).
 * Billing logic will be re-enabled when RuStore maven is back online.
 */
@CapacitorPlugin(name = "RuStoreBilling")
class RuStoreBillingPlugin : Plugin() {
    @PluginMethod
    fun purchase(call: PluginCall) {
        call.reject("RuStore Pay SDK temporarily unavailable")
    }
}
