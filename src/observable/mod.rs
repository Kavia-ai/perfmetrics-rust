use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::any::Any;
use std::error::Error;
use std::fmt;

pub mod metrics;

/// Error types for observer operations
#[derive(Debug)]
pub enum ObserverError {
    DuplicateObserver(String),
    ObserverNotFound(String),
    NotificationError(String),
}

impl fmt::Display for ObserverError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ObserverError::DuplicateObserver(id) => write!(f, "Observer with ID {} already exists", id),
            ObserverError::ObserverNotFound(id) => write!(f, "Observer with ID {} not found", id),
            ObserverError::NotificationError(msg) => write!(f, "Failed to notify observer: {}", msg),
        }
    }
}

impl Error for ObserverError {}

/// Trait for objects that can receive metric updates
pub trait Observer: Send + Sync {
    // PUBLIC_INTERFACE
    fn on_metric_update(&self, metric: Box<dyn Any + Send>) -> Result<(), Box<dyn Error>>;

    // PUBLIC_INTERFACE
    fn get_id(&self) -> String;
}

/// Trait for objects that can be observed for metric updates
pub trait Observable: Send + Sync {
    // PUBLIC_INTERFACE
    fn add_observer(&mut self, observer: Arc<dyn Observer>) -> Result<(), ObserverError>;

    // PUBLIC_INTERFACE
    fn remove_observer(&mut self, observer_id: &str) -> Result<(), ObserverError>;

    // PUBLIC_INTERFACE
    fn notify_observers(&self, metric: Box<dyn Any + Send>) -> Vec<Result<(), Box<dyn Error>>>;
}

/// Default implementation of Observable trait that can be used by metric collectors
pub struct DefaultObservable {
    observers: Arc<Mutex<HashMap<String, Arc<dyn Observer>>>>,
}

impl DefaultObservable {
    // PUBLIC_INTERFACE
    pub fn new() -> Self {
        DefaultObservable {
            observers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Observable for DefaultObservable {
    fn add_observer(&mut self, observer: Arc<dyn Observer>) -> Result<(), ObserverError> {
        let mut observers = self.observers.lock().unwrap();
        let id = observer.get_id();

        if observers.contains_key(&id) {
            return Err(ObserverError::DuplicateObserver(id));
        }

        observers.insert(id, observer);
        Ok(())
    }

    fn remove_observer(&mut self, observer_id: &str) -> Result<(), ObserverError> {
        let mut observers = self.observers.lock().unwrap();

        if observers.remove(observer_id).is_none() {
            return Err(ObserverError::ObserverNotFound(observer_id.to_string()));
        }

        Ok(())
    }

    fn notify_observers(&self, _metric: Box<dyn Any + Send>) -> Vec<Result<(), Box<dyn Error>>> {
        Vec::new() // Return empty vector, effectively disabling notifications
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
    fn test_add_remove_observer() {
        let mut observable = DefaultObservable::new();
        let was_notified = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(TestObserver {
            id: "test".to_string(),
            was_notified: was_notified.clone(),
        });

        // Test adding observer
        assert!(observable.add_observer(observer.clone()).is_ok());

        // Test duplicate observer
        assert!(matches!(
            observable.add_observer(observer.clone()),
            Err(ObserverError::DuplicateObserver(_))
        ));

        // Test removing observer
        assert!(observable.remove_observer("test").is_ok());

        // Test removing non-existent observer
        assert!(matches!(
            observable.remove_observer("test"),
            Err(ObserverError::ObserverNotFound(_))
        ));
    }

    #[test]
    fn test_notify_observers() {
        let mut observable = DefaultObservable::new();
        let was_notified = Arc::new(AtomicBool::new(false));
        let observer = Arc::new(TestObserver {
            id: "test".to_string(),
            was_notified: was_notified.clone(),
        });

        observable.add_observer(observer).unwrap();

        let metric = Box::new(42u32);
        let results = observable.notify_observers(metric);

        assert!(results.iter().all(|r| r.is_ok()));
        assert!(was_notified.load(Ordering::SeqCst));
    }
}
