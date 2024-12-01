#include "browser_metric_handler.h"

namespace WPEFramework {
namespace Plugin {

BrowserMetricHandler::BrowserMetricHandler() = default;
BrowserMetricHandler::~BrowserMetricHandler() = default;

void BrowserMetricHandler::RegisterCallback(const std::string& id, std::shared_ptr<MetricCallback> callback) {
    std::lock_guard<std::mutex> lock(_mutex);
    _callbacks[id] = callback;
}

void BrowserMetricHandler::UnregisterCallback(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mutex);
    _callbacks.erase(id);
}

void BrowserMetricHandler::NotifyLoadFinished(const std::string& url, int32_t httpStatus, bool success) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& [id, callback] : _callbacks) {
        if (callback) {
            callback->OnLoadFinished(url, httpStatus, success);
        }
    }
}

void BrowserMetricHandler::NotifyURLChange(const std::string& url, bool loaded) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& [id, callback] : _callbacks) {
        if (callback) {
            callback->OnURLChange(url, loaded);
        }
    }
}

void BrowserMetricHandler::NotifyVisibilityChange(bool hidden) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& [id, callback] : _callbacks) {
        if (callback) {
            callback->OnVisibilityChange(hidden);
        }
    }
}

void BrowserMetricHandler::NotifyPageClosure() {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& [id, callback] : _callbacks) {
        if (callback) {
            callback->OnPageClosure();
        }
    }
}

} // namespace Plugin
} // namespace WPEFramework