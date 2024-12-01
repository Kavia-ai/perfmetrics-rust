use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::traits::{IBasicMetricsLogger, IStateMetricsLogger, IBrowserMetricsLogger};

/// Represents the main PerformanceMetrics plugin that implements metrics logging interfaces
/// 
/// This struct provides a concrete implementation of the metrics logging traits,
/// managing metric collection and reporting for the RDK services framework.
pub struct PerformanceMetricsPlugin {
    enabled: Arc<Mutex<bool>>,
    state: Arc<Mutex<PluginState>>,
}

/// Represents the internal state of the plugin
#[derive(Debug, Clone, Copy)]
enum PluginState {
    Running,
    Suspended,
}

impl Default for PerformanceMetricsPlugin {
    fn default() -> Self {
        Self {
            enabled: Arc::new(Mutex::new(false)),
            state: Arc::new(Mutex::new(PluginState::Suspended)),
        }
    }
}

impl IBasicMetricsLogger for PerformanceMetricsPlugin {
    fn enable(&mut self) -> Result<(), Box<dyn Error>> {
        let mut enabled = self.enabled.lock().map_err(|e| format!("Failed to acquire lock: {}", e))?;
        *enabled = true;
        Ok(())
    }

    fn disable(&mut self) -> Result<(), Box<dyn Error>> {
        let mut enabled = self.enabled.lock().map_err(|e| format!("Failed to acquire lock: {}", e))?;
        *enabled = false;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.lock().map(|guard| *guard).unwrap_or(false)
    }

    fn log_counter(&self, name: &str, value: i64) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual counter logging
        println!("Counter metric - {}: {}", name, value);
        Ok(())
    }

    fn log_gauge(&self, name: &str, value: f64) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual gauge logging
        println!("Gauge metric - {}: {}", name, value);
        Ok(())
    }

    fn log_timing(&self, name: &str, duration_ms: u64) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual timing logging
        println!("Timing metric - {}: {}ms", name, duration_ms);
        Ok(())
    }
}

impl IStateMetricsLogger for PerformanceMetricsPlugin {
    fn resumed(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        let mut state = self.state.lock().map_err(|e| format!("Failed to acquire lock: {}", e))?;
        *state = PluginState::Running;
        // TODO: Implement actual state transition logging
        println!("State transition: Resumed");
        Ok(())
    }

    fn suspended(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        let mut state = self.state.lock().map_err(|e| format!("Failed to acquire lock: {}", e))?;
        *state = PluginState::Suspended;
        // TODO: Implement actual state transition logging
        println!("State transition: Suspended");
        Ok(())
    }
}

impl IBrowserMetricsLogger for PerformanceMetricsPlugin {
    fn load_finished(
        &mut self,
        url: String,
        http_status: u16,
        success: bool,
        total_success: u32,
        total_failed: u32
    ) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual page load metrics logging
        println!("Page load finished - URL: {}, Status: {}, Success: {}", url, http_status, success);
        println!("Total loads - Success: {}, Failed: {}", total_success, total_failed);
        Ok(())
    }

    fn url_change(
        &mut self,
        url: String,
        loaded: bool
    ) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual URL change logging
        println!("URL changed - New URL: {}, Loaded: {}", url, loaded);
        Ok(())
    }

    fn visibility_change(
        &mut self,
        hidden: bool
    ) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual visibility change logging
        println!("Visibility changed - Hidden: {}", hidden);
        Ok(())
    }

    fn page_closure(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.is_enabled() {
            return Err("Metrics logging is disabled".into());
        }
        // TODO: Implement actual page closure logging
        println!("Page closed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enable_disable() {
        let mut plugin = PerformanceMetricsPlugin::default();
        assert!(!plugin.is_enabled());
        
        plugin.enable().unwrap();
        assert!(plugin.is_enabled());
        
        plugin.disable().unwrap();
        assert!(!plugin.is_enabled());
    }

    #[test]
    fn test_state_transitions() {
        let mut plugin = PerformanceMetricsPlugin::default();
        plugin.enable().unwrap();
        
        plugin.resumed().unwrap();
        plugin.suspended().unwrap();
    }

    #[test]
    fn test_browser_metrics() {
        let mut plugin = PerformanceMetricsPlugin::default();
        plugin.enable().unwrap();
        
        plugin.load_finished(
            "https://example.com".to_string(),
            200,
            true,
            1,
            0
        ).unwrap();
        
        plugin.url_change(
            "https://example.com/page".to_string(),
            true
        ).unwrap();
        
        plugin.visibility_change(false).unwrap();
        plugin.page_closure().unwrap();
    }
}