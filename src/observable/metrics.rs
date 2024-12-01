use std::any::Any;
use std::error::Error;
use std::sync::{Arc, Mutex};
use super::{Observable, Observer, DefaultObservable};

/// Basic metric type for simple numeric values
#[derive(Debug, Clone)]
pub struct BasicMetric {
    name: String,
    value: f64,
}

impl BasicMetric {
    pub fn new(name: String, value: f64) -> Self {
        BasicMetric { name, value }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }
}

/// State metric type for tracking state changes
#[derive(Debug, Clone)]
pub struct StateMetric {
    name: String,
    state: String,
    timestamp: u64,
}

impl StateMetric {
    pub fn new(name: String, state: String, timestamp: u64) -> Self {
        StateMetric { name, state, timestamp }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_state(&self) -> &str {
        &self.state
    }

    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }
}

/// Browser metric type for tracking browser-specific metrics
#[derive(Debug, Clone)]
pub struct BrowserMetric {
    name: String,
    url: String,
    load_time: u64,
    memory_usage: u64,
}

impl BrowserMetric {
    pub fn new(name: String, url: String, load_time: u64, memory_usage: u64) -> Self {
        BrowserMetric {
            name,
            url,
            load_time,
            memory_usage,
        }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_url(&self) -> &str {
        &self.url
    }

    pub fn get_load_time(&self) -> u64 {
        self.load_time
    }

    pub fn get_memory_usage(&self) -> u64 {
        self.memory_usage
    }
}

/// Observable implementation for basic metrics
pub struct BasicMetricObservable {
    observable: DefaultObservable,
    current_metric: Arc<Mutex<Option<BasicMetric>>>,
}

impl BasicMetricObservable {
    pub fn new() -> Self {
        BasicMetricObservable {
            observable: DefaultObservable::new(),
            current_metric: Arc::new(Mutex::new(None)),
        }
    }

    pub fn update_metric(&mut self, metric: BasicMetric) -> Vec<Result<(), Box<dyn Error>>> {
        {
            let mut current = self.current_metric.lock().unwrap();
            *current = Some(metric.clone());
        }
        self.observable.notify_observers(Box::new(metric))
    }
}

impl Observable for BasicMetricObservable {
    fn add_observer(&mut self, observer: Arc<dyn Observer>) -> Result<(), super::ObserverError> {
        self.observable.add_observer(observer)
    }

    fn remove_observer(&mut self, observer_id: &str) -> Result<(), super::ObserverError> {
        self.observable.remove_observer(observer_id)
    }

    fn notify_observers(&self, metric: Box<dyn Any + Send>) -> Vec<Result<(), Box<dyn Error>>> {
        self.observable.notify_observers(metric)
    }
}

/// Observable implementation for state metrics
pub struct StateMetricObservable {
    observable: DefaultObservable,
    current_metric: Arc<Mutex<Option<StateMetric>>>,
}

impl StateMetricObservable {
    pub fn new() -> Self {
        StateMetricObservable {
            observable: DefaultObservable::new(),
            current_metric: Arc::new(Mutex::new(None)),
        }
    }

    pub fn update_metric(&mut self, metric: StateMetric) -> Vec<Result<(), Box<dyn Error>>> {
        {
            let mut current = self.current_metric.lock().unwrap();
            *current = Some(metric.clone());
        }
        self.observable.notify_observers(Box::new(metric))
    }
}

impl Observable for StateMetricObservable {
    fn add_observer(&mut self, observer: Arc<dyn Observer>) -> Result<(), super::ObserverError> {
        self.observable.add_observer(observer)
    }

    fn remove_observer(&mut self, observer_id: &str) -> Result<(), super::ObserverError> {
        self.observable.remove_observer(observer_id)
    }

    fn notify_observers(&self, metric: Box<dyn Any + Send>) -> Vec<Result<(), Box<dyn Error>>> {
        self.observable.notify_observers(metric)
    }
}

/// Observable implementation for browser metrics
pub struct BrowserMetricObservable {
    observable: DefaultObservable,
    current_metric: Arc<Mutex<Option<BrowserMetric>>>,
}

impl BrowserMetricObservable {
    pub fn new() -> Self {
        BrowserMetricObservable {
            observable: DefaultObservable::new(),
            current_metric: Arc::new(Mutex::new(None)),
        }
    }

    pub fn update_metric(&mut self, metric: BrowserMetric) -> Vec<Result<(), Box<dyn Error>>> {
        {
            let mut current = self.current_metric.lock().unwrap();
            *current = Some(metric.clone());
        }
        self.observable.notify_observers(Box::new(metric))
    }
}

impl Observable for BrowserMetricObservable {
    fn add_observer(&mut self, observer: Arc<dyn Observer>) -> Result<(), super::ObserverError> {
        self.observable.add_observer(observer)
    }

    fn remove_observer(&mut self, observer_id: &str) -> Result<(), super::ObserverError> {
        self.observable.remove_observer(observer_id)
    }

    fn notify_observers(&self, metric: Box<dyn Any + Send>) -> Vec<Result<(), Box<dyn Error>>> {
        self.observable.notify_observers(metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct TestObserver {
        id: String,
        was_notified: Arc<AtomicBool>,
    }

    impl Observer for TestObserver {
        fn on_metric_update(&self, _metric: Box<dyn Any + Send>) -> Result<(), Box<dyn Error>> {
            self.was_notified.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn get_id(&self) -> String {
            self.id.clone()
        }
    }

    #[test]
    fn test_basic_metric_observable() {
        let mut observable = BasicMetricObservable::new();
        let was_notified = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(TestObserver {
            id: "test".to_string(),
            was_notified: was_notified.clone(),
        });

        observable.add_observer(observer).unwrap();
        
        let metric = BasicMetric::new("test_metric".to_string(), 42.0);
        let results = observable.update_metric(metric);
        
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(was_notified.load(Ordering::SeqCst));
    }

    #[test]
    fn test_state_metric_observable() {
        let mut observable = StateMetricObservable::new();
        let was_notified = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(TestObserver {
            id: "test".to_string(),
            was_notified: was_notified.clone(),
        });

        observable.add_observer(observer).unwrap();
        
        let metric = StateMetric::new("test_state".to_string(), "active".to_string(), 123456);
        let results = observable.update_metric(metric);
        
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(was_notified.load(Ordering::SeqCst));
    }

    #[test]
    fn test_browser_metric_observable() {
        let mut observable = BrowserMetricObservable::new();
        let was_notified = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(TestObserver {
            id: "test".to_string(),
            was_notified: was_notified.clone(),
        });

        observable.add_observer(observer).unwrap();
        
        let metric = BrowserMetric::new(
            "test_browser".to_string(),
            "https://example.com".to_string(),
            100,
            1024,
        );
        let results = observable.update_metric(metric);
        
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(was_notified.load(Ordering::SeqCst));
    }
}