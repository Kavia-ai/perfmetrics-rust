#pragma once

#include <mutex>
#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

namespace WPEFramework {
namespace Plugin {

/**
 * @brief Handler for browser-specific metrics with thread-safe callback handling
 */
class BrowserMetricHandler {
public:
    // PUBLIC_INTERFACE
    struct MetricCallback {
        virtual ~MetricCallback() = default;
        virtual void OnLoadFinished(const std::string& url, int32_t httpStatus, bool success) = 0;
        virtual void OnURLChange(const std::string& url, bool loaded) = 0;
        virtual void OnVisibilityChange(bool hidden) = 0;
        virtual void OnPageClosure() = 0;
    };

    // PUBLIC_INTERFACE
    BrowserMetricHandler();
    ~BrowserMetricHandler();

    // PUBLIC_INTERFACE
    void RegisterCallback(const std::string& id, std::shared_ptr<MetricCallback> callback);
    void UnregisterCallback(const std::string& id);
    
    // PUBLIC_INTERFACE
    void NotifyLoadFinished(const std::string& url, int32_t httpStatus, bool success);
    void NotifyURLChange(const std::string& url, bool loaded);
    void NotifyVisibilityChange(bool hidden);
    void NotifyPageClosure();

private:
    std::mutex _mutex;
    std::unordered_map<std::string, std::shared_ptr<MetricCallback>> _callbacks;
};

} // namespace Plugin
} // namespace WPEFramework