use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{RwLock, Arc, atomic::{AtomicU64, Ordering}};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::mpsc::{channel, Sender, Receiver};
use regex::Regex;
use std::any::Any;
use crate::observable::{Observable, Observer, ObserverError, DefaultObservable};

mod validation;
use validation::{Validator, ValidationRule, ValidationContext, ValidationResult, RangeRule, PatternRule, NonEmptyRule};
use crate::config::validation::PerformanceMetricsRule;

/// Configuration change event type
#[derive(Debug, Clone)]
pub enum ConfigChangeEvent {
    BrowserMetricsUpdate(BrowserMetricConfig),
    FullConfigUpdate(Config),
}

/// Configuration change listener type
pub type ConfigChangeListener = Arc<dyn Fn(ConfigChangeEvent) + Send + Sync>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BrowserMetricConfig {
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Maximum number of metrics to store in memory
    pub max_metrics_count: usize,
    /// List of browser metrics to collect
    pub metrics: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    /// Browser metrics configuration
    pub browser_metrics: BrowserMetricConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            browser_metrics: BrowserMetricConfig {
                sampling_interval_ms: 1000,
                max_metrics_count: 1000,
                metrics: vec![
                    "memory_usage".to_string(),
                    "cpu_usage".to_string(),
                    "fps".to_string(),
                ],
            },
        }
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::thread;
    use std::time::{Duration, Instant};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::collections::VecDeque;
    use std::sync::mpsc::channel;
    use rand::{thread_rng, Rng};

    /// Performance metrics for stress tests
    #[derive(Debug, Default)]
    pub(crate) struct PerformanceMetrics {
        operation_latencies: Vec<Duration>,
        memory_samples: Vec<usize>,
        cpu_samples: Vec<f64>,
        total_operations: usize,
        start_time: Option<Instant>,
        pub(crate) peak_memory: usize,
        pub(crate) peak_cpu: f64,
    }

    impl PerformanceMetrics {
        pub(crate) fn new() -> Self {
            PerformanceMetrics {
                start_time: Some(Instant::now()),
                ..Default::default()
            }
        }

        pub(crate) fn record_operation(&mut self, latency: Duration) {
            self.operation_latencies.push(latency);
            self.total_operations += 1;
        }

        pub(crate) fn record_memory(&mut self, memory: usize) {
            self.memory_samples.push(memory);
            self.peak_memory = self.peak_memory.max(memory);
        }

        pub(crate) fn record_cpu(&mut self, cpu: f64) {
            self.cpu_samples.push(cpu);
            self.peak_cpu = self.peak_cpu.max(cpu);
        }

        pub(crate) fn calculate_percentile(&self, percentile: f64) -> Duration {
            if self.operation_latencies.is_empty() {
                return Duration::from_secs(0);
            }
            let mut sorted_latencies = self.operation_latencies.clone();
            sorted_latencies.sort();
            let index = ((sorted_latencies.len() as f64 * percentile) / 100.0) as usize;
            sorted_latencies[index.min(sorted_latencies.len() - 1)]
        }

        pub(crate) fn print_report(&self) {
            let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
            println!("\nPerformance Test Report");
            println!("----------------------");
            println!("Total operations: {}", self.total_operations);
            println!("Total duration: {:?}", elapsed);
            println!("Operations/sec: {:.2}", self.total_operations as f64 / elapsed.as_secs_f64());
            
            if !self.operation_latencies.is_empty() {
                println!("\nLatency Statistics:");
                println!("  p50: {:?}", self.calculate_percentile(50.0));
                println!("  p90: {:?}", self.calculate_percentile(90.0));
                println!("  p99: {:?}", self.calculate_percentile(99.0));
            }

            if !self.memory_samples.is_empty() {
                let avg_memory = self.memory_samples.iter().sum::<usize>() / self.memory_samples.len();
                println!("\nMemory Statistics:");
                println!("  Peak: {} bytes", self.peak_memory);
                println!("  Average: {} bytes", avg_memory);
            }

            if !self.cpu_samples.is_empty() {
                let avg_cpu = self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len() as f64;
                println!("\nCPU Statistics:");
                println!("  Peak: {:.2}%", self.peak_cpu);
                println!("  Average: {:.2}%", avg_cpu);
            }
        }
    }

    /// Helper function to monitor memory usage of the current process
    pub(crate) fn get_memory_usage() -> usize {
        // This is a basic implementation that reads /proc/self/statm on Linux
        // Returns 0 on unsupported platforms
        if let Ok(contents) = fs::read_to_string("/proc/self/statm") {
            if let Some(value) = contents.split_whitespace().next() {
                if let Ok(pages) = value.parse::<usize>() {
                    return pages * 4096; // Convert pages to bytes (assuming 4KB pages)
                }
            }
        }
        0
    }

    /// Helper function to monitor CPU usage of the current process
    pub(crate) fn get_cpu_usage() -> f64 {
        // This is a basic implementation that reads /proc/self/stat on Linux
        // Returns 0.0 on unsupported platforms
        if let Ok(contents) = fs::read_to_string("/proc/self/stat") {
            let fields: Vec<&str> = contents.split_whitespace().collect();
            if fields.len() >= 14 {
                // Fields 13 and 14 are utime and stime
                if let (Ok(utime), Ok(stime)) = (fields[13].parse::<u64>(), fields[14].parse::<u64>()) {
                    let total_time = utime + stime;
                    // Convert to percentage (assuming system hz is 100)
                    return (total_time as f64) / 100.0;
                }
            }
        }
        0.0
    }

    #[test]
    fn test_concurrent_readers_writers() {
        const NUM_READERS: usize = 50;
        const NUM_WRITERS: usize = 10;
        const TEST_DURATION_SECS: u64 = 5;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        let total_reads = Arc::new(AtomicUsize::new(0));
        let total_writes = Arc::new(AtomicUsize::new(0));
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(get_cpu_usage());
        
        // Spawn reader threads
        let mut reader_handles = vec![];
        for _ in 0..NUM_READERS {
            let manager_clone = Arc::clone(&manager);
            let total_reads_clone = Arc::clone(&total_reads);
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                let start = Instant::now();
                while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
                    let op_start = Instant::now();
                    if manager_clone.get_config().is_ok() {
                        total_reads_clone.fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut metrics) = metrics_clone.write() {
                            metrics.record_operation(op_start.elapsed());
                        }
                    }
                    thread::sleep(Duration::from_micros(100));
                }
            });
            reader_handles.push(handle);
        }

        // Spawn writer threads
        let mut writer_handles = vec![];
        for i in 0..NUM_WRITERS {
            let manager_clone = Arc::clone(&manager);
            let total_writes_clone = Arc::clone(&total_writes);
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                let start = Instant::now();
                while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
                    let op_start = Instant::now();
                    let config = Config {
                        browser_metrics: BrowserMetricConfig {
                            sampling_interval_ms: 1000 + i as u64,
                            max_metrics_count: 1000 + i,
                            metrics: vec!["memory_usage".to_string()],
                        },
                    };
                    if manager_clone.update_config(config).is_ok() {
                        total_writes_clone.fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut metrics) = metrics_clone.write() {
                            metrics.record_operation(op_start.elapsed());
                        }
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            });
            writer_handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in reader_handles {
            handle.join().unwrap();
        }
        for handle in writer_handles {
            handle.join().unwrap();
        }

        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(get_memory_usage());
        metrics.record_cpu(get_cpu_usage());
        
        let total_reads = total_reads.load(Ordering::Relaxed);
        let total_writes = total_writes.load(Ordering::Relaxed);
        
        println!("Stress test results:");
        println!("Total reads: {}", total_reads);
        println!("Total writes: {}", total_writes);
        
        // Print detailed performance metrics
        metrics.print_report();
        
        // Assertions
        assert!(total_reads > 0, "Expected some successful reads");
        assert!(total_writes > 0, "Expected some successful writes");
        assert!(metrics.peak_memory < 10 * 1024 * 1024, "Excessive memory usage detected");
        assert!(metrics.calculate_percentile(99.0) < Duration::from_millis(10), "Operation latency too high");
    }

    #[test]
    fn test_rapid_config_updates() {
        const NUM_UPDATES: usize = 10000;
        const BATCH_SIZE: usize = 100;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(get_cpu_usage());
        
        // Perform rapid updates in batches
        for batch in 0..(NUM_UPDATES / BATCH_SIZE) {
            let batch_start = Instant::now();
            let metrics_clone = Arc::clone(&metrics);
            
            for i in 0..BATCH_SIZE {
                let op_start = Instant::now();
                let update_num = batch * BATCH_SIZE + i;
                let config = Config {
                    browser_metrics: BrowserMetricConfig {
                        sampling_interval_ms: 1000 + update_num as u64,
                        max_metrics_count: 1000 + update_num,
                        metrics: vec!["memory_usage".to_string(), "cpu_usage".to_string()],
                    },
                };
                manager.update_config(config).unwrap();
                
                if let Ok(mut metrics) = metrics_clone.write() {
                    metrics.record_operation(op_start.elapsed());
                }
            }
            
            // Record resource usage after each batch
            if let Ok(mut metrics) = metrics_clone.write() {
                metrics.record_memory(get_memory_usage());
                metrics.record_cpu(get_cpu_usage());
            }
            
            thread::sleep(Duration::from_millis(10)); // Small delay between batches
        }
        
        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(get_memory_usage());
        metrics.record_cpu(get_cpu_usage());
        
        println!("Rapid update test results:");
        println!("Total updates: {}", NUM_UPDATES);
        
        // Print detailed performance metrics
        metrics.print_report();
        
        // Performance assertions
        assert!(metrics.calculate_percentile(90.0) < Duration::from_millis(10), "Operation latency too high");
        assert!(metrics.peak_memory < 5 * 1024 * 1024, "Excessive memory usage detected");
        assert!(metrics.peak_cpu < 100.0, "CPU usage too high");
    }

    #[test]
    fn test_concurrent_validation_stress() {
        const NUM_THREADS: usize = 20;
        const TEST_DURATION_SECS: u64 = 5;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        let validation_errors = Arc::new(AtomicUsize::new(0));
        let successful_validations = Arc::new(AtomicUsize::new(0));
        
        let mut handles = vec![];
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(get_cpu_usage());
        
        // Spawn threads that will attempt to validate configurations concurrently
        for _ in 0..NUM_THREADS {
            let manager_clone = Arc::clone(&manager);
            let metrics_clone = Arc::clone(&metrics);
            let validation_errors_clone = Arc::clone(&validation_errors);
            let successful_validations_clone = Arc::clone(&successful_validations);
            
            let handle = thread::spawn(move || {
                let mut rng = thread_rng();
                let start = Instant::now();
                
                while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
                    let op_start = Instant::now();
                    
                    // Generate random configuration (sometimes invalid)
                    let config = Config {
                        browser_metrics: BrowserMetricConfig {
                            sampling_interval_ms: if rng.gen_bool(0.3) { 0 } else { rng.gen_range(1..10000) },
                            max_metrics_count: if rng.gen_bool(0.3) { 0 } else { rng.gen_range(1..10000) },
                            metrics: if rng.gen_bool(0.3) {
                                vec![]
                            } else {
                                vec!["memory_usage".to_string(), "cpu_usage".to_string()]
                            },
                        },
                    };
                    
                    // Attempt to update configuration
                    match manager_clone.update_config(config) {
                        Ok(_) => {
                            successful_validations_clone.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            validation_errors_clone.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    
                    if let Ok(mut metrics) = metrics_clone.write() {
                        metrics.record_operation(op_start.elapsed());
                    }
                    
                    thread::sleep(Duration::from_micros(100));
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(get_memory_usage());
        metrics.record_cpu(get_cpu_usage());
        
        let total_validations = successful_validations.load(Ordering::Relaxed);
        let total_errors = validation_errors.load(Ordering::Relaxed);
        
        println!("Concurrent validation stress test results:");
        println!("Total successful validations: {}", total_validations);
        println!("Total validation errors: {}", total_errors);
        
        // Print detailed performance metrics
        metrics.print_report();
        
        // Assertions
        assert!(total_validations > 0, "Expected some successful validations");
        assert!(total_errors > 0, "Expected some validation errors due to invalid configs");
        assert!(metrics.calculate_percentile(99.0) < Duration::from_millis(10), "Validation latency too high");
    }

    #[test]
    fn test_validation_error_handling_stress() {
        const NUM_ITERATIONS: usize = 1000;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        let mut error_types = std::collections::HashMap::new();
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(get_cpu_usage());
        
        let mut rng = thread_rng();
        
        for i in 0..NUM_ITERATIONS {
            let op_start = Instant::now();
            
            // Generate different types of invalid configurations
            let config = match i % 4 {
                0 => Config {
                    browser_metrics: BrowserMetricConfig {
                        sampling_interval_ms: 0, // Invalid interval
                        max_metrics_count: 1000,
                        metrics: vec!["memory_usage".to_string()],
                    },
                },
                1 => Config {
                    browser_metrics: BrowserMetricConfig {
                        sampling_interval_ms: 1000,
                        max_metrics_count: 0, // Invalid count
                        metrics: vec!["memory_usage".to_string()],
                    },
                },
                2 => Config {
                    browser_metrics: BrowserMetricConfig {
                        sampling_interval_ms: 1000,
                        max_metrics_count: 1000,
                        metrics: vec![], // Empty metrics
                    },
                },
                3 => Config {
                    browser_metrics: BrowserMetricConfig {
                        sampling_interval_ms: rng.gen_range(1..10000),
                        max_metrics_count: rng.gen_range(1..10000),
                        metrics: vec!["invalid_metric".to_string()], // Valid but unused metric
                    },
                },
                _ => unreachable!(),
            };
            
            // Attempt to update and categorize errors
            match manager.update_config(config) {
                Ok(_) => {
                    *error_types.entry("success".to_string()).or_insert(0) += 1;
                }
                Err(e) => {
                    *error_types.entry(e).or_insert(0) += 1;
                }
            }
            
            if let Ok(mut metrics) = metrics_clone.write() {
                metrics.record_operation(op_start.elapsed());
            }
            
            // Periodically record resource usage
            if i % 100 == 0 {
                if let Ok(mut metrics) = metrics_clone.write() {
                    metrics.record_memory(get_memory_usage());
                    metrics.record_cpu(get_cpu_usage());
                }
            }
        }
        
        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(get_memory_usage());
        metrics.record_cpu(get_cpu_usage());
        
        println!("Validation error handling stress test results:");
        println!("Error type distribution:");
        for (error_type, count) in error_types.iter() {
            println!("  {}: {}", error_type, count);
        }
        
        // Print detailed performance metrics
        metrics.print_report();
        
        // Assertions
        assert!(error_types.len() >= 3, "Expected at least 3 different types of validation errors");
        assert!(metrics.calculate_percentile(99.0) < Duration::from_millis(10), "Error handling latency too high");
        assert!(metrics.peak_memory < 10 * 1024 * 1024, "Excessive memory usage during error handling");
    }

    #[test]
    fn test_config_update_performance_under_load() {
        const NUM_ITERATIONS: usize = 1000;
        const NUM_LISTENERS: usize = 100;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::new()));
        let (tx, rx) = channel();
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(get_cpu_usage());
        
        // Register many listeners
        for i in 0..NUM_LISTENERS {
            let tx = tx.clone();
            manager.register_listener(Arc::new(move |event| {
                tx.send((i, event)).unwrap();
            })).unwrap();
        }
        
        // Perform updates while measuring time
        for i in 0..NUM_ITERATIONS {
            let op_start = Instant::now();
            let metrics_clone = Arc::clone(&metrics);
            
            let config = Config {
                browser_metrics: BrowserMetricConfig {
                    sampling_interval_ms: 1000 + i as u64,
                    max_metrics_count: 1000 + i,
                    metrics: vec!["memory_usage".to_string()],
                },
            };
            
            manager.update_config(config).unwrap();
            
            // Record operation latency
            if let Ok(mut metrics) = metrics_clone.write() {
                metrics.record_operation(op_start.elapsed());
            }
            
            // Periodically record resource usage
            if i % 100 == 0 {
                if let Ok(mut metrics) = metrics_clone.write() {
                    metrics.record_memory(get_memory_usage());
                    metrics.record_cpu(get_cpu_usage());
                }
            }
            
            // Drain notifications
            while rx.try_recv().is_ok() {}
        }
        
        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(get_memory_usage());
        metrics.record_cpu(get_cpu_usage());
        
        println!("Performance test results:");
        println!("Total iterations: {}", NUM_ITERATIONS);
        
        // Print detailed performance metrics
        metrics.print_report();
        
        // Performance assertions
        assert!(metrics.calculate_percentile(90.0) < Duration::from_millis(10), "Operation latency too high");
        assert!(metrics.peak_memory < 10 * 1024 * 1024, "Excessive memory usage detected");
        assert!(metrics.peak_cpu < 100.0, "CPU usage too high");
    }
}

/// Thread-safe configuration container with change notification support
pub struct ConfigManager {
    config: RwLock<Config>,
    listeners: RwLock<Vec<ConfigChangeListener>>,
    version: AtomicU64,
    metric_name_pattern: Regex,
    observable: RwLock<DefaultObservable>,
    validator: RwLock<Validator>,
}

impl ConfigManager {
    /// Creates a new ConfigManager with default configuration
    pub fn new() -> Self {
        let mut validator = Validator::new();
        
        // Add validation rules for browser metrics
        // Add performance metrics validation rule
        validator.add_rule(Arc::new(PerformanceMetricsRule::new()
            .with_sample_rate_bounds(0.1, 1000.0)
            .with_buffer_size_bounds(10, 1000000)
            .with_max_metric_name_length(128)
            .with_max_metrics_per_category(1000)));

        // Add basic validation rules
        validator.add_rule(Arc::new(RangeRule::new(1, u64::MAX, "sampling_interval_ms".to_string())));
        validator.add_rule(Arc::new(RangeRule::new(1, usize::MAX, "max_metrics_count".to_string())));
        validator.add_rule(Arc::new(NonEmptyRule::new("metrics".to_string())));
        validator.add_rule(Arc::new(PatternRule::new(
            Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$").unwrap(),
            "metric_name".to_string()
        )));
        
        ConfigManager {
            config: RwLock::new(Config::default()),
            listeners: RwLock::new(Vec::new()),
            version: AtomicU64::new(1),
            metric_name_pattern: Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$").unwrap(),
            observable: RwLock::new(DefaultObservable::new()),
            validator: RwLock::new(validator),
        }
    }

    /// Registers a configuration observer
    /// 
    /// # Arguments
    /// * `observer` - The observer to register
    /// 
    /// # Returns
    /// * `Ok(())` if observer was registered successfully
    /// * `Err(ObserverError)` if registration failed
    pub fn register_observer(&self, observer: Arc<dyn Observer>) -> Result<(), ObserverError> {
        self.observable.write().unwrap().add_observer(observer)
    }

    /// Unregisters a configuration observer
    /// 
    /// # Arguments
    /// * `observer_id` - ID of the observer to unregister
    /// 
    /// # Returns
    /// * `Ok(())` if observer was unregistered successfully
    /// * `Err(ObserverError)` if unregistration failed
    pub fn unregister_observer(&self, observer_id: &str) -> Result<(), ObserverError> {
        self.observable.write().unwrap().remove_observer(observer_id)
    }

    /// Notifies observers about configuration changes
    fn notify_config_observers(&self, config: &Config) {
        let metric = Box::new(config.clone());
        self.observable.read().unwrap().notify_observers(metric);
    }

    /// Gets the current configuration version
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Validates a metric name against the allowed pattern
    fn validate_metric_name(&self, name: &str) -> Result<(), String> {
        if !self.metric_name_pattern.is_match(name) {
            return Err(format!(
                "Invalid metric name '{}'. Must start with a letter and contain only letters, numbers, and underscores (max 64 chars)",
                name
            ));
        }
        Ok(())
    }

    /// Registers a listener for configuration changes
    /// 
    /// # Arguments
    /// * `listener` - Callback function that will be called when configuration changes
    /// 
    /// # Returns
    /// * `Ok(())` if listener was registered successfully
    /// * `Err(String)` if registration failed
    pub fn register_listener(&self, listener: ConfigChangeListener) -> Result<(), String> {
        match self.listeners.write() {
            Ok(mut listeners) => {
                listeners.push(listener);
                Ok(())
            }
            Err(e) => Err(format!("Failed to register listener: {}", e))
        }
    }

    /// Notifies all registered listeners about a configuration change
    fn notify_listeners(&self, event: ConfigChangeEvent) -> Result<(), String> {
        match self.listeners.read() {
            Ok(listeners) => {
                for listener in listeners.iter() {
                    listener(event.clone());
                }
                Ok(())
            }
            Err(e) => Err(format!("Failed to notify listeners: {}", e))
        }
    }

    /// Validates a configuration update
    fn validate_config(&self, config: &Config) -> Result<(), String> {
        match self.validator.read() {
            Ok(validator) => {
                validator.validate(config, "config").map_err(|error| {
                    format!("{}", error.path)
                })
            }
            Err(e) => Err(format!("Failed to acquire validator lock: {}", e))
        }
    }

    /// Loads and validates configuration from a JSON file
    /// 
    /// # Arguments
    /// * `path` - Path to the configuration file
    /// 
    /// # Returns
    /// * `Ok(())` if configuration was loaded successfully
    /// * `Err(String)` if configuration could not be loaded or validation failed
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        // Verify file exists and is readable
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(format!("Configuration file does not exist: {}", path_ref.display()));
        }
        if !path_ref.is_file() {
            return Err(format!("Path is not a file: {}", path_ref.display()));
        }

        // Read file contents
        let mut file = File::open(path_ref)
            .map_err(|e| format!("Failed to open config file: {}", e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        // Parse and validate JSON structure
        let json_value: Value = serde_json::from_str(&contents)
            .map_err(|e| format!("Invalid JSON format: {}", e))?;

        // Validate required fields
        if !json_value.is_object() {
            return Err("Configuration must be a JSON object".to_string());
        }
        
        let obj = json_value.as_object().unwrap();
        if !obj.contains_key("browser_metrics") {
            return Err("Missing required field 'browser_metrics'".to_string());
        }

        // Parse into Config struct
        let new_config: Config = serde_json::from_value(json_value)
            .map_err(|e| format!("Invalid configuration structure: {}", e))?;

        // Validate configuration
        self.validate_config(&new_config)?;

        // Update configuration
        let mut config = self.config.write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;
        *config = new_config;
        // Increment version
        self.version.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Gets a copy of the current configuration
    /// 
    /// # Returns
    /// * `Ok(Config)` if configuration was retrieved successfully
    /// * `Err(String)` if configuration could not be retrieved
    pub fn get_config(&self) -> Result<Config, String> {
        self.config.read()
            .map(|config| config.clone())
            .map_err(|e| format!("Failed to acquire read lock: {}", e))
    }

    /// Updates the configuration with validation and notification
    /// 
    /// # Arguments
    /// * `new_config` - New configuration to set
    /// 
    /// # Returns
    /// * `Ok(())` if configuration was updated successfully
    /// * `Err(String)` if configuration update failed
    pub fn update_config(&self, new_config: Config) -> Result<(), String> {
        // Validate the new configuration
        self.validate_config(&new_config)?;

        // Update configuration with write lock
        match self.config.write() {
            Ok(mut config) => {
                *config = new_config.clone();
                // Increment version
                self.version.fetch_add(1, Ordering::Release);
                // Notify listeners and observers about the update
                self.notify_listeners(ConfigChangeEvent::FullConfigUpdate(new_config.clone()))?;
                self.notify_config_observers(&new_config);
                Ok(())
            }
            Err(e) => Err(format!("Failed to acquire write lock: {}", e))
        }
    }

    /// Updates only the browser metrics configuration
    /// 
    /// # Arguments
    /// * `new_metrics_config` - New browser metrics configuration
    /// 
    /// # Returns
    /// * `Ok(())` if configuration was updated successfully
    /// * `Err(String)` if configuration update failed
    pub fn update_browser_metrics(&self, new_metrics_config: BrowserMetricConfig) -> Result<(), String> {
        // Validate the new metrics configuration
        let test_config = Config {
            browser_metrics: new_metrics_config.clone(),
        };
        self.validate_config(&test_config)?;

        // Update configuration with write lock
        match self.config.write() {
            Ok(mut config) => {
                config.browser_metrics = new_metrics_config.clone();
                // Increment version
                self.version.fetch_add(1, Ordering::Release);
                // Notify listeners and observers about the update
                self.notify_listeners(ConfigChangeEvent::BrowserMetricsUpdate(new_metrics_config.clone()))?;
                let updated_config = self.get_config()?;
                self.notify_config_observers(&updated_config);
                Ok(())
            }
            Err(e) => Err(format!("Failed to acquire write lock: {}", e))
        }
    }

    /// Saves the current configuration to a file
    /// 
    /// # Arguments
    /// * `path` - Path to save the configuration file
    /// 
    /// # Returns
    /// * `Ok(())` if configuration was saved successfully
    /// * `Err(String)` if configuration could not be saved
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let config = match self.config.read() {
            Ok(config) => config,
            Err(e) => return Err(format!("Failed to acquire read lock: {}", e))
        };

        let json = serde_json::to_string_pretty(&*config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(path, json)
            .map_err(|e| format!("Failed to write config file: {}", e))
    }

    /// Gets a reference to the configuration with read-only access
    /// 
    /// # Returns
    /// * `Ok(&Config)` if lock was acquired successfully
    /// * `Err(String)` if lock could not be acquired
    pub fn get_config_ref(&self) -> Result<std::sync::RwLockReadGuard<Config>, String> {
        self.config.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};
    use rand::{thread_rng, Rng};

    #[test]
    fn test_default_config() {
        let manager = ConfigManager::new();
        let config = manager.get_config().unwrap();
        assert_eq!(config.browser_metrics.sampling_interval_ms, 1000);
        assert_eq!(config.browser_metrics.max_metrics_count, 1000);
        assert_eq!(config.browser_metrics.metrics.len(), 3);
    }

    #[test]
    fn test_load_and_save_config() -> Result<(), String> {
        let manager = ConfigManager::new();
        let mut temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
        
        // Create test config
        let test_config = r#"{
            "browser_metrics": {
                "sampling_interval_ms": 2000,
                "max_metrics_count": 500,
                "metrics": ["memory_usage", "cpu_usage"]
            }
        }"#;
        
        temp_file.write_all(test_config.as_bytes()).map_err(|e| e.to_string())?;
        
        // Load config
        manager.load_from_file(temp_file.path())?;
        let loaded_config = manager.get_config().unwrap();
        
        assert_eq!(loaded_config.browser_metrics.sampling_interval_ms, 2000);
        assert_eq!(loaded_config.browser_metrics.max_metrics_count, 500);
        assert_eq!(loaded_config.browser_metrics.metrics.len(), 2);
        
        // Save config to new file
        let save_file = NamedTempFile::new().map_err(|e| e.to_string())?;
        manager.save_to_file(save_file.path())?;
        
        // Verify saved content
        let mut saved_content = String::new();
        File::open(save_file.path())
            .map_err(|e| e.to_string())?
            .read_to_string(&mut saved_content)
            .map_err(|e| e.to_string())?;
        
        let saved_config: Config = serde_json::from_str(&saved_content)
            .map_err(|e| format!("Failed to parse saved config: {}", e))?;
            
        assert_eq!(saved_config.browser_metrics.sampling_interval_ms, 2000);
        assert_eq!(saved_config.browser_metrics.max_metrics_count, 500);
        assert_eq!(saved_config.browser_metrics.metrics.len(), 2);
        
        Ok(())
    }

    #[test]
    fn test_config_update_with_notification() -> Result<(), String> {
        let manager = ConfigManager::new();
        let (tx, rx) = std::sync::mpsc::channel();
        
        // Create a test listener
        let listener = Arc::new(move |event: ConfigChangeEvent| {
            tx.send(event).unwrap();
        });
        
        // Register the listener
        manager.register_listener(listener)?;
        
        // Update browser metrics configuration
        let new_metrics = BrowserMetricConfig {
            sampling_interval_ms: 3000,
            max_metrics_count: 750,
            metrics: vec!["memory_usage".to_string()],
        };
        
        manager.update_browser_metrics(new_metrics.clone())?;
        
        // Verify the notification was received
        match rx.recv() {
            Ok(ConfigChangeEvent::BrowserMetricsUpdate(updated_metrics)) => {
                assert_eq!(updated_metrics.sampling_interval_ms, 3000);
                assert_eq!(updated_metrics.max_metrics_count, 750);
                assert_eq!(updated_metrics.metrics.len(), 1);
            },
            _ => return Err("Expected BrowserMetricsUpdate event".to_string()),
        }
        
        // Update full configuration
        let new_config = Config {
            browser_metrics: BrowserMetricConfig {
                sampling_interval_ms: 4000,
                max_metrics_count: 1500,
                metrics: vec!["cpu_usage".to_string()],
            },
        };
        
        manager.update_config(new_config.clone())?;
        
        // Verify the notification was received
        match rx.recv() {
            Ok(ConfigChangeEvent::FullConfigUpdate(updated_config)) => {
                assert_eq!(updated_config.browser_metrics.sampling_interval_ms, 4000);
                assert_eq!(updated_config.browser_metrics.max_metrics_count, 1500);
                assert_eq!(updated_config.browser_metrics.metrics.len(), 1);
            },
            _ => return Err("Expected FullConfigUpdate event".to_string()),
        }
        
        Ok(())
    }

    #[test]
    fn test_config_validation() {
        let manager = ConfigManager::new();
        
        // Test invalid sampling interval
        let invalid_interval = BrowserMetricConfig {
            sampling_interval_ms: 0,
            max_metrics_count: 1000,
            metrics: vec!["memory_usage".to_string()],
        };
        assert!(manager.update_browser_metrics(invalid_interval).is_err());
        
        // Test invalid metrics count
        let invalid_count = BrowserMetricConfig {
            sampling_interval_ms: 1000,
            max_metrics_count: 0,
            metrics: vec!["memory_usage".to_string()],
        };
        assert!(manager.update_browser_metrics(invalid_count).is_err());
        
        // Test empty metrics list
        let empty_metrics = BrowserMetricConfig {
            sampling_interval_ms: 1000,
            max_metrics_count: 1000,
            metrics: vec![],
        };
        assert!(manager.update_browser_metrics(empty_metrics).is_err());

        // Test extremely large values
        let large_values = BrowserMetricConfig {
            sampling_interval_ms: u64::MAX,
            max_metrics_count: usize::MAX,
            metrics: vec!["memory_usage".to_string()],
        };
        // Large values should be valid
        assert!(manager.update_browser_metrics(large_values).is_ok());

        // Test duplicate metrics
        let duplicate_metrics = BrowserMetricConfig {
            sampling_interval_ms: 1000,
            max_metrics_count: 1000,
            metrics: vec!["memory_usage".to_string(), "memory_usage".to_string()],
        };
        // Duplicates are currently allowed by design
        assert!(manager.update_browser_metrics(duplicate_metrics).is_ok());
    }

    #[test]
    fn test_concurrent_updates() {
        use std::thread;
        use std::sync::Arc;

        let manager = Arc::new(ConfigManager::new());
        let manager_clone = Arc::clone(&manager);

        // Create a channel to track updates
        let (tx, rx) = channel();
        let tx_clone = tx.clone();

        // Register listeners for both full and metric updates
        manager.register_listener(Arc::new(move |event| {
            tx.send(event).unwrap();
        })).unwrap();

        // Spawn a thread to update browser metrics
        let metrics_thread = thread::spawn(move || {
            for i in 1..=5 {
                let metrics_config = BrowserMetricConfig {
                    sampling_interval_ms: i * 1000,
                    max_metrics_count: (i * 100) as usize,
                    metrics: vec!["memory_usage".to_string()],
                };
                manager_clone.update_browser_metrics(metrics_config).unwrap();
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });

        // Update full config in main thread
        for i in 1..=5 {
            let config = Config {
                browser_metrics: BrowserMetricConfig {
                    sampling_interval_ms: i * 2000,
                    max_metrics_count: (i * 200) as usize,
                    metrics: vec!["cpu_usage".to_string()],
                },
            };
            manager.update_config(config).unwrap();
            thread::sleep(std::time::Duration::from_millis(10));
        }

        metrics_thread.join().unwrap();

        // Verify we received all updates (10 total: 5 metric updates + 5 full updates)
        let mut update_count = 0;
        while let Ok(_) = rx.try_recv() {
            update_count += 1;
        }
        assert_eq!(update_count, 10);
    }

    #[test]
    fn test_config_update_validation_and_rollback() {
        let manager = ConfigManager::new();
        let original_config = manager.get_config().unwrap();

        // Attempt to update with invalid config
        let invalid_config = Config {
            browser_metrics: BrowserMetricConfig {
                sampling_interval_ms: 0, // Invalid value
                max_metrics_count: 1000,
                metrics: vec!["memory_usage".to_string()],
            },
        };

        // Update should fail
        assert!(manager.update_config(invalid_config).is_err());

        // Verify config remains unchanged
        let current_config = manager.get_config().unwrap();
        assert_eq!(
            current_config.browser_metrics.sampling_interval_ms,
            original_config.browser_metrics.sampling_interval_ms
        );
    }

    #[test]
    fn test_multiple_listeners() {
        let manager = ConfigManager::new();
        let mut received_updates: Vec<ConfigChangeEvent> = vec![];
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();

        // Register multiple listeners
        manager.register_listener(Arc::new(move |event| {
            tx1.send(event).unwrap();
        })).unwrap();

        manager.register_listener(Arc::new(move |event| {
            tx2.send(event).unwrap();
        })).unwrap();

        // Perform update
        let new_config = BrowserMetricConfig {
            sampling_interval_ms: 2000,
            max_metrics_count: 500,
            metrics: vec!["memory_usage".to_string()],
        };
        manager.update_browser_metrics(new_config).unwrap();

        // Verify both listeners received the update
        assert!(rx1.recv().is_ok());
        assert!(rx2.recv().is_ok());
    }

    #[test]
    fn test_validation_edge_cases_stress() {
        const NUM_ITERATIONS: usize = 1000;
        const MAX_STRING_LENGTH: usize = 1_000_000;
        const MAX_METRICS: usize = 100_000;
        
        let manager = Arc::new(ConfigManager::new());
        let metrics = Arc::new(RwLock::new(stress_tests::PerformanceMetrics::new()));
        
        // Initial measurements
        let metrics_clone = Arc::clone(&metrics);
        metrics_clone.write().unwrap().record_memory(stress_tests::get_memory_usage());
        metrics_clone.write().unwrap().record_cpu(stress_tests::get_cpu_usage());
        
        let mut validation_results = std::collections::HashMap::new();
        
        // Create a pool of threads for concurrent validation
        let pool_size = num_cpus::get();
        let (tx, rx) = channel();
        
        for _ in 0..pool_size {
            let manager_clone = Arc::clone(&manager);
            let tx_clone = tx.clone();
            let metrics_clone = Arc::clone(&metrics);
            
            thread::spawn(move || {
                let mut rng = thread_rng();

                for i in 0..NUM_ITERATIONS / pool_size {
                    let op_start = Instant::now();
                    
                    // Generate edge case configurations
                    let config = match i % 8 {
                        0 => Config {
                            // Test minimum valid values
                            browser_metrics: BrowserMetricConfig {
                                sampling_interval_ms: 1,
                                max_metrics_count: 1,
                                metrics: vec!["minimal".to_string()],
                            },
                        },
                        1 => Config {
                            // Test maximum numeric values
                            browser_metrics: BrowserMetricConfig {
                                sampling_interval_ms: u64::MAX,
                                max_metrics_count: usize::MAX,
                                metrics: vec!["maximal".to_string()],
                            },
                        },
                        2 => {
                            // Test extremely long metric names
                            let long_metric = "x".repeat(MAX_STRING_LENGTH);
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: 1000,
                                    max_metrics_count: 1000,
                                    metrics: vec![long_metric],
                                },
                            }
                        },
                        3 => {
                            // Test large number of metrics
                            let metrics: Vec<String> = (0..MAX_METRICS)
                                .map(|i| format!("metric_{}", i))
                                .collect();
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: 1000,
                                    max_metrics_count: MAX_METRICS,
                                    metrics,
                                },
                            }
                        },
                        4 => {
                            // Test Unicode characters in metric names
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: 1000,
                                    max_metrics_count: 1000,
                                    metrics: vec![
                                        "метрика".to_string(),
                                        "度量".to_string(),
                                        "μέτρηση".to_string(),
                                    ],
                                },
                            }
                        },
                        5 => {
                            // Test special characters in metric names
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: 1000,
                                    max_metrics_count: 1000,
                                    metrics: vec![
                                        "!@#$%^&*()".to_string(),
                                        "metric/with/slashes".to_string(),
                                        "metric.with.dots".to_string(),
                                    ],
                                },
                            }
                        },
                        6 => {
                            // Test whitespace in metric names
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: 1000,
                                    max_metrics_count: 1000,
                                    metrics: vec![
                                        "   leading spaces".to_string(),
                                        "trailing spaces   ".to_string(),
                                        "   both   ends   ".to_string(),
                                        "\t\ttabs\t\t".to_string(),
                                        "\n\nnewlines\n\n".to_string(),
                                    ],
                                },
                            }
                        },
                        7 => {
                            // Test random combinations
                            let random_interval = if rng.gen_bool(0.5) { 1 } else { u64::MAX };
                            let random_count = if rng.gen_bool(0.5) { 1 } else { usize::MAX };
                            let random_metrics = if rng.gen_bool(0.5) {
                                vec!["random".to_string()]
                            } else {
                                (0..rng.gen_range(1..1000))
                                    .map(|i| format!("random_metric_{}", i))
                                    .collect()
                            };
                            
                            Config {
                                browser_metrics: BrowserMetricConfig {
                                    sampling_interval_ms: random_interval,
                                    max_metrics_count: random_count,
                                    metrics: random_metrics,
                                },
                            }
                        },
                        _ => unreachable!(),
                    };
                    
                    // Attempt to update configuration
                    let result = manager_clone.update_config(config.clone());
                    tx_clone.send((i % 8, result.is_ok())).unwrap();
                    
                    // Record performance metrics
                    if let Ok(mut metrics) = metrics_clone.write() {
                        metrics.record_operation(op_start.elapsed());
                        
                        // Periodically record resource usage
                        if i % 100 == 0 {
                            metrics.record_memory(stress_tests::get_memory_usage());
                            metrics.record_cpu(stress_tests::get_cpu_usage());
                        }
                    }
                }
            });
        }
        
        // Collect results
        drop(tx); // Drop original sender so we know when all threads are done
        while let Ok((case_type, success)) = rx.recv() {
            let entry = validation_results.entry(case_type).or_insert((0, 0));
            if success {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
        
        // Final measurements and report
        let metrics_clone = Arc::clone(&metrics);
        let mut metrics = metrics_clone.write().unwrap();
        metrics.record_memory(stress_tests::get_memory_usage());
        metrics.record_cpu(stress_tests::get_cpu_usage());
        
        println!("\nEdge Case Validation Results:");
        println!("----------------------------");
        for (case_type, (successes, failures)) in validation_results.iter() {
            let case_name = match case_type {
                0 => "Minimum valid values",
                1 => "Maximum numeric values",
                2 => "Extremely long metric names",
                3 => "Large number of metrics",
                4 => "Unicode characters",
                5 => "Special characters",
                6 => "Whitespace handling",
                7 => "Random combinations",
                _ => "Unknown",
            };
            println!("{}: {} successes, {} failures", case_name, successes, failures);
        }
        
        // Print performance metrics
        metrics.print_report();
        
        // Assertions
        assert!(metrics.calculate_percentile(99.0) < Duration::from_millis(100), 
            "Edge case validation latency too high");
        assert!(metrics.peak_memory < 100 * 1024 * 1024, 
            "Excessive memory usage during edge case validation");
        assert!(metrics.peak_cpu < 100.0, 
            "CPU usage too high during edge case validation");
        
        // Verify that certain cases were handled as expected
        for (case_type, (successes, failures)) in validation_results.iter() {
            match case_type {
                0 => assert!(*successes > 0, "Minimum valid values should be accepted"),
                1 => assert!(*successes > 0, "Maximum numeric values should be accepted"),
                2..=6 => assert!(*successes + *failures > 0,
                                 "Edge case type {} should have been tested", case_type),
                7 => assert!(*failures > 0,
                             "Random combinations should have produced some failures"),
                _ => unreachable!(),
            }
        }
    }
}
