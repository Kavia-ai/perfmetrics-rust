use std::any::Any;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::observable::metrics::{BasicMetric, StateMetric, BrowserMetric};
use crate::observable::Observer;

/// Error types for metric handlers
#[derive(Debug)]
pub enum MetricHandlerError {
    InvalidMetricType(String),
    HandlerError(String),
}

impl fmt::Display for MetricHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MetricHandlerError::InvalidMetricType(msg) => write!(f, "Invalid metric type: {}", msg),
            MetricHandlerError::HandlerError(msg) => write!(f, "Handler error: {}", msg),
        }
    }
}

impl Error for MetricHandlerError {}

/// Handler for basic numeric metrics
pub struct BasicMetricHandler {
    id: String,
    callback: Box<dyn Fn(&BasicMetric) -> Result<(), Box<dyn Error>> + Send + Sync>,
}

impl BasicMetricHandler {
    // PUBLIC_INTERFACE
    pub fn new<F>(id: String, callback: F) -> Self
    where
        F: Fn(&BasicMetric) -> Result<(), Box<dyn Error>> + Send + Sync + 'static,
    {
        BasicMetricHandler {
            id,
            callback: Box::new(callback),
        }
    }
}

impl Observer for BasicMetricHandler {
    fn on_metric_update(&self, metric: Box<dyn Any + Send>) -> Result<(), Box<dyn Error>> {
        match metric.downcast_ref::<BasicMetric>() {
            Some(basic_metric) => (self.callback)(basic_metric),
            None => Err(Box::new(MetricHandlerError::InvalidMetricType(
                "Expected BasicMetric".to_string(),
            ))),
        }
    }

    fn get_id(&self) -> String {
        self.id.clone()
    }
}

/// Handler for state change metrics
pub struct StateMetricHandler {
    id: String,
    callback: Box<dyn Fn(&StateMetric) -> Result<(), Box<dyn Error>> + Send + Sync>,
}

impl StateMetricHandler {
    // PUBLIC_INTERFACE
    pub fn new<F>(id: String, callback: F) -> Self
    where
        F: Fn(&StateMetric) -> Result<(), Box<dyn Error>> + Send + Sync + 'static,
    {
        StateMetricHandler {
            id,
            callback: Box::new(callback),
        }
    }
}

impl Observer for StateMetricHandler {
    fn on_metric_update(&self, metric: Box<dyn Any + Send>) -> Result<(), Box<dyn Error>> {
        match metric.downcast_ref::<StateMetric>() {
            Some(state_metric) => (self.callback)(state_metric),
            None => Err(Box::new(MetricHandlerError::InvalidMetricType(
                "Expected StateMetric".to_string(),
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
    fn test_basic_metric_handler() {
        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = was_called.clone();
        
        let handler = BasicMetricHandler::new("test_basic".to_string(), move |metric| {
            assert_eq!(metric.get_name(), "test_metric");
            assert_eq!(metric.get_value(), 42.0);
            was_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        });

        let metric = Box::new(BasicMetric::new("test_metric".to_string(), 42.0));
        assert!(handler.on_metric_update(metric).is_ok());
        assert!(was_called.load(Ordering::SeqCst));

        // Test invalid metric type
        let invalid_metric = Box::new(StateMetric::new(
            "test_state".to_string(),
            "active".to_string(),
            123456,
        ));
        assert!(handler.on_metric_update(invalid_metric).is_err());
    }

    #[test]
    fn test_state_metric_handler() {
        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = was_called.clone();
        
        let handler = StateMetricHandler::new("test_state".to_string(), move |metric| {
            assert_eq!(metric.get_name(), "test_state");
            assert_eq!(metric.get_state(), "active");
            assert_eq!(metric.get_timestamp(), 123456);
            was_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        });

        let metric = Box::new(StateMetric::new(
            "test_state".to_string(),
            "active".to_string(),
            123456,
        ));
        assert!(handler.on_metric_update(metric).is_ok());
        assert!(was_called.load(Ordering::SeqCst));

        // Test invalid metric type
        let invalid_metric = Box::new(BasicMetric::new("test_metric".to_string(), 42.0));
        assert!(handler.on_metric_update(invalid_metric).is_err());
    }
}

mod browser_metric_handler;

pub use browser_metric_handler::BrowserMetricHandler;
