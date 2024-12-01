// WPEFramework interface header for PerformanceMetrics
#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

void* performance_metrics_create(void);
void performance_metrics_destroy(void* handle);
const char* performance_metrics_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_METRICS_H