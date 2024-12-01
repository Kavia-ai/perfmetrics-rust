use std::any::Any;
use std::error::Error;
use std::sync::{Arc, Mutex};

use crate::observable::metrics::BrowserMetric;
use crate::observable::Observer;
use super::MetricHandlerError;

/// Handler for browser-specific metrics with thread-safe callback handling
pub struct BrowserMetricHandler {
    id: String,
    callback: Arc<Mutex<Box<dyn Fn(&BrowserMetric) -> Result<(), Box<dyn Error>> + Send + Sync>>>,
}

impl BrowserMetricHandler {
    // PUBLIC_INTERFACE
    /// Creates a new BrowserMetricHandler with the specified ID and callback function
    /// 
    /// # Arguments
    /// * `id` - Unique identifier for the handler
    /// * `callback` - Function to be called when a browser metric is received
    pub fn new<F>(id: String, callback: F) -> Self
    where
        F: Fn(&BrowserMetric) -> Result<(), Box<dyn Error>> + Send + Sync + 'static,
    {
        BrowserMetricHandler {
            id,
            callback: Arc::new(Mutex::new(Box::new(callback))),
        }
    }

    // PUBLIC_INTERFACE
    /// Updates the callback function for this handler
    /// 
    /// # Arguments
    /// * `new_callback` - New callback function to replace the existing one
    pub fn update_callback<F>(&self, new_callback: F) -> Result<(), MetricHandlerError>
    where
        F: Fn(&BrowserMetric) -> Result<(), Box<dyn Error>> + Send + Sync + 'static,
    {
        match self.callback.lock() {
            Ok(mut guard) => {
                *guard = Box::new(new_callback);
                Ok(())
            }
            Err(_) => Err(MetricHandlerError::HandlerError(
                "Failed to acquire lock for callback update".to_string(),
            )),
        }
    }
}

impl Observer for BrowserMetricHandler {
    fn on_metric_update(&self, metric: Box<dyn Any + Send>) -> Result<(), Box<dyn Error>> {
        match metric.downcast_ref::<BrowserMetric>() {
            Some(browser_metric) => {
                match self.callback.lock() {
                    Ok(callback) => callback(browser_metric),
                    Err(_) => Err(Box::new(MetricHandlerError::HandlerError(
                        "Failed to acquire lock for callback execution".to_string(),
                    ))),
                }
            }
            None => Err(Box::new(MetricHandlerError::InvalidMetricType(
                "Expected BrowserMetric".to_string(),
            ))),
        }
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_browser_metric_handler() {
        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = was_called.clone();
        
        let handler = BrowserMetricHandler::new("test_browser".to_string(), move |metric| {
            assert_eq!(metric.get_name(), "test_browser");
            assert_eq!(metric.get_url(), "https://example.com");
            assert_eq!(metric.get_load_time(), 100);
            assert_eq!(metric.get_memory_usage(), 1024);
            was_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        });

        let metric = Box::new(BrowserMetric::new(
            "test_browser".to_string(),
            "https://example.com".to_string(),
            100,
            1024,
        ));
        assert!(handler.on_metric_update(metric).is_ok());
        assert!(was_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_invalid_metric_type() {
        let handler = BrowserMetricHandler::new("test_browser".to_string(), |_| Ok(()));
        let invalid_metric = Box::new(42u32);
        assert!(matches!(
            handler.on_metric_update(invalid_metric).unwrap_err().downcast_ref::<MetricHandlerError>(),
            Some(MetricHandlerError::InvalidMetricType(_))
        ));
    }

    #[test]
    fn test_update_callback() {
        let first_called = Arc::new(AtomicBool::new(false));
        let second_called = Arc::new(AtomicBool::new(false));
        
        let first_called_clone = first_called.clone();
        let second_called_clone = second_called.clone();

        let handler = BrowserMetricHandler::new("test_browser".to_string(), move |_| {
            first_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        });

        let metric = Box::new(BrowserMetric::new(
            "test_browser".to_string(),
            "https://example.com".to_string(),
            100,
            1024,
        ));

        // Test first callback
        assert!(handler.on_metric_update(metric.clone()).is_ok());
        assert!(first_called.load(Ordering::SeqCst));
        assert!(!second_called.load(Ordering::SeqCst));

        // Update callback
        assert!(handler.update_callback(move |_| {
            second_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        }).is_ok());

        // Test second callback
        assert!(handler.on_metric_update(metric).is_ok());
        assert!(second_called.load(Ordering::SeqCst));
    }
}