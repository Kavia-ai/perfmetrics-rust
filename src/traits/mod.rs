use std::error::Error;

/// Represents a basic metrics logging interface for general purpose metrics
/// 
/// This trait provides functionality for basic metrics logging with enable/disable capabilities.
/// Implementations should ensure thread-safety for the enabled state.
/// 
/// # PUBLIC_INTERFACE
pub trait IBasicMetricsLogger {
    /// Enable metrics logging
    /// 
    /// # Returns
    /// - `Ok(())` if the operation was successful
    /// - `Err` if enabling metrics logging failed
    fn enable(&mut self) -> Result<(), Box<dyn Error>>;
    
    /// Disable metrics logging
    /// 
    /// # Returns
    /// - `Ok(())` if the operation was successful
    /// - `Err` if disabling metrics logging failed
    fn disable(&mut self) -> Result<(), Box<dyn Error>>;
    
    /// Check if metrics logging is enabled
    /// 
    /// # Returns
    /// - `true` if metrics logging is enabled
    /// - `false` if metrics logging is disabled
    fn is_enabled(&self) -> bool;
    
    /// Log a simple counter metric
    /// 
    /// # Arguments
    /// * `name` - The name of the metric
    /// * `value` - The counter value to log
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn log_counter(&self, name: &str, value: i64) -> Result<(), Box<dyn Error>>;
    
    /// Log a gauge metric that can go up or down
    /// 
    /// # Arguments
    /// * `name` - The name of the metric
    /// * `value` - The gauge value to log
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn log_gauge(&self, name: &str, value: f64) -> Result<(), Box<dyn Error>>;
    
    /// Log a timing metric in milliseconds
    /// 
    /// # Arguments
    /// * `name` - The name of the metric
    /// * `duration_ms` - The duration in milliseconds to log
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn log_timing(&self, name: &str, duration_ms: u64) -> Result<(), Box<dyn Error>>;
}

/// Represents a state metrics logging interface for tracking state changes
/// 
/// This trait extends IBasicMetricsLogger to add state change tracking capabilities.
/// Implementations should handle state transitions between resumed and suspended states.
/// 
/// # PUBLIC_INTERFACE
pub trait IStateMetricsLogger: IBasicMetricsLogger {
    /// Log a state transition to resumed state
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn resumed(&mut self) -> Result<(), Box<dyn Error>>;
    
    /// Log a state transition to suspended state
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn suspended(&mut self) -> Result<(), Box<dyn Error>>;
}

/// Represents a browser-specific metrics logging interface that extends state metrics logging
/// 
/// This trait extends IStateMetricsLogger to add browser-specific metrics logging capabilities.
/// It provides methods for tracking page loads, URL changes, visibility changes, and page closures.
/// 
/// # PUBLIC_INTERFACE
pub trait IBrowserMetricsLogger: IStateMetricsLogger {
    /// Log when a page load has finished
    /// 
    /// # Arguments
    /// * `url` - The URL of the loaded page
    /// * `http_status` - The HTTP status code of the response
    /// * `success` - Whether the page load was successful
    /// * `total_success` - Total number of successful page loads
    /// * `total_failed` - Total number of failed page loads
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn load_finished(
        &mut self,
        url: String,
        http_status: u16,
        success: bool,
        total_success: u32,
        total_failed: u32
    ) -> Result<(), Box<dyn Error>>;

    /// Log when the URL changes
    /// 
    /// # Arguments
    /// * `url` - The new URL
    /// * `loaded` - Whether the page is fully loaded
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn url_change(
        &mut self,
        url: String,
        loaded: bool
    ) -> Result<(), Box<dyn Error>>;

    /// Log when the page visibility changes
    /// 
    /// # Arguments
    /// * `hidden` - Whether the page is hidden
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn visibility_change(
        &mut self,
        hidden: bool
    ) -> Result<(), Box<dyn Error>>;

    /// Log when a page is closed
    /// 
    /// # Returns
    /// - `Ok(())` if logging was successful
    /// - `Err` if logging failed or metrics are disabled
    fn page_closure(&mut self) -> Result<(), Box<dyn Error>>;
}
