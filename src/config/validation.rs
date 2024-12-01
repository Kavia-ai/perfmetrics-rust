use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use serde::Serialize;
use lazy_static::lazy_static;
use std::error::Error;
use std::fmt;
use std::sync::Arc as StdArc;

/// Performance metrics validation rule
pub struct PerformanceMetricsRule {
    min_sample_rate: f64,
    max_sample_rate: f64,
    min_buffer_size: usize,
    max_buffer_size: usize,
    max_metric_name_length: usize,
    max_metrics_per_category: usize,
}

impl PerformanceMetricsRule {
    pub fn new() -> Self {
        PerformanceMetricsRule {
            min_sample_rate: 0.1,  // Minimum 0.1 Hz sampling
            max_sample_rate: 1000.0, // Maximum 1000 Hz sampling
            min_buffer_size: 10,    // Minimum 10 samples buffer
            max_buffer_size: 1000000, // Maximum 1M samples buffer
            max_metric_name_length: 128, // Maximum metric name length
            max_metrics_per_category: 1000, // Maximum metrics per category
        }
    }

    pub fn with_sample_rate_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_sample_rate = min;
        self.max_sample_rate = max;
        self
    }

    pub fn with_buffer_size_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_buffer_size = min;
        self.max_buffer_size = max;
        self
    }

    pub fn with_max_metric_name_length(mut self, max: usize) -> Self {
        self.max_metric_name_length = max;
        self
    }

    pub fn with_max_metrics_per_category(mut self, max: usize) -> Self {
        self.max_metrics_per_category = max;
        self
    }
}

impl ValidationRule for PerformanceMetricsRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize performance metrics: {}", e)),
            path: path.to_string(),
        })?;

        let obj = value.as_object().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Performance metrics must be an object".to_string()),
            path: path.to_string(),
        })?;

        let mut errors = Vec::new();

        // Validate sample rate
        if let Some(sample_rate) = obj.get("sample_rate") {
            if let Some(rate) = sample_rate.as_f64() {
                if rate < self.min_sample_rate || rate > self.max_sample_rate {
                    errors.push(ValidationError {
                        kind: ValidationErrorKind::RuleFailed(
                            format!("Sample rate must be between {} and {} Hz", 
                                self.min_sample_rate, self.max_sample_rate)
                        ),
                        path: format!("{}.sample_rate", path),
                    });
                }
            } else {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::TypeMismatch("Sample rate must be a number".to_string()),
                    path: format!("{}.sample_rate", path),
                });
            }
        }

        // Validate buffer size
        if let Some(buffer_size) = obj.get("buffer_size") {
            if let Some(size) = buffer_size.as_u64() {
                let size = size as usize;
                if size < self.min_buffer_size || size > self.max_buffer_size {
                    errors.push(ValidationError {
                        kind: ValidationErrorKind::RuleFailed(
                            format!("Buffer size must be between {} and {}", 
                                self.min_buffer_size, self.max_buffer_size)
                        ),
                        path: format!("{}.buffer_size", path),
                    });
                }
            } else {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::TypeMismatch("Buffer size must be a positive integer".to_string()),
                    path: format!("{}.buffer_size", path),
                });
            }
        }

        // Validate metrics configuration
        if let Some(metrics) = obj.get("metrics") {
            if let Some(metrics_obj) = metrics.as_object() {
                // Validate each metric category
                for (category, metrics_list) in metrics_obj {
                    if let Some(metrics_arr) = metrics_list.as_array() {
                        // Check number of metrics in category
                        if metrics_arr.len() > self.max_metrics_per_category {
                            errors.push(ValidationError {
                                kind: ValidationErrorKind::RuleFailed(
                                    format!("Category '{}' exceeds maximum of {} metrics", 
                                        category, self.max_metrics_per_category)
                                ),
                                path: format!("{}.metrics.{}", path, category),
                            });
                        }

                        // Validate each metric name
                        for (idx, metric) in metrics_arr.iter().enumerate() {
                            if let Some(name) = metric.as_str() {
                                if name.len() > self.max_metric_name_length {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Metric name exceeds maximum length of {} characters", 
                                                self.max_metric_name_length)
                                        ),
                                        path: format!("{}.metrics.{}.{}", path, category, idx),
                                    });
                                }

                                // Validate metric name pattern
                                let name_pattern = regex::Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]*$").unwrap();
                                if !name_pattern.is_match(name) {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Metric name '{}' must start with a letter and contain only alphanumeric characters and underscores", 
                                                name)
                                        ),
                                        path: format!("{}.metrics.{}.{}", path, category, idx),
                                    });
                                }
                            } else {
                                errors.push(ValidationError {
                                    kind: ValidationErrorKind::TypeMismatch("Metric name must be a string".to_string()),
                                    path: format!("{}.metrics.{}.{}", path, category, idx),
                                });
                            }
                        }
                    } else {
                        errors.push(ValidationError {
                            kind: ValidationErrorKind::TypeMismatch(
                                format!("Metrics for category '{}' must be an array", category)
                            ),
                            path: format!("{}.metrics.{}", path, category),
                        });
                    }
                }
            } else {
                errors.push(ValidationError {
                    kind: ValidationErrorKind::TypeMismatch("Metrics must be an object of arrays".to_string()),
                    path: format!("{}.metrics", path),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationError {
                kind: ValidationErrorKind::AggregateError(errors),
                path: path.to_string(),
            })
        }
    }

    fn description(&self) -> String {
        format!(
            "Performance metrics must have valid sample rate ({}-{} Hz), buffer size ({}-{}), \
            and properly formatted metric names (max length: {})",
            self.min_sample_rate, self.max_sample_rate,
            self.min_buffer_size, self.max_buffer_size,
            self.max_metric_name_length
        )
    }
}

/// Severity level for validation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Warning - validation issue that doesn't prevent operation but should be addressed
    Warning,
    /// Error - validation issue that must be fixed
    Error,
    /// Critical - severe validation issue that could cause system instability
    Critical,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationSeverity::Warning => write!(f, "WARNING"),
            ValidationSeverity::Error => write!(f, "ERROR"),
            ValidationSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Metadata about the field being validated
#[derive(Debug, Clone)]
pub struct ValidationFieldMetadata {
    /// Name of the field
    pub field_name: String,
    /// Expected type of the field
    pub expected_type: String,
    /// Description of valid values
    pub valid_values: String,
    /// Whether the field is required
    pub required: bool,
}

impl ValidationFieldMetadata {
    pub fn new(field_name: &str, expected_type: &str, valid_values: &str, required: bool) -> Self {
        ValidationFieldMetadata {
            field_name: field_name.to_string(),
            expected_type: expected_type.to_string(),
            valid_values: valid_values.to_string(),
            required: required,
        }
    }
}

/// Custom error type for validation failures
#[derive(Debug)]
pub enum ValidationErrorKind {
    /// Rule validation failure with severity level
    RuleFailed(String, ValidationSeverity),
    /// Serialization error with context
    SerializationError(String, Option<Box<dyn Error + Send + Sync>>),
    /// Type mismatch with expected type information
    TypeMismatch(String, String), // (message, expected_type)
    /// Registry error with context
    RegistryError(String, Option<Box<dyn Error + Send + Sync>>),
    /// Multiple validation errors
    AggregateError(Vec<ValidationError>),
}

impl fmt::Display for ValidationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationErrorKind::RuleFailed(msg, severity) => {
                write!(f, "[{}] Validation rule failed: {}", severity, msg)
            }
            ValidationErrorKind::SerializationError(msg, source) => {
                write!(f, "Serialization error: {}", msg)?;
                if let Some(err) = source {
                    write!(f, " (Caused by: {})", err)?;
                }
                Ok(())
            }
            ValidationErrorKind::TypeMismatch(msg, expected) => {
                write!(f, "Type mismatch: {} (Expected: {})", msg, expected)
            }
            ValidationErrorKind::RegistryError(msg, source) => {
                write!(f, "Registry error: {}", msg)?;
                if let Some(err) = source {
                    write!(f, " (Caused by: {})", err)?;
                }
                Ok(())
            }
            ValidationErrorKind::AggregateError(errors) => {
                writeln!(f, "Multiple validation errors occurred:")?;
                for error in errors {
                    writeln!(f, "  - {}", error)?;
                }
                Ok(())
            }
        }
    }
}

impl Error for ValidationErrorKind {}

/// Validation error containing detailed error information
#[derive(Debug)]
pub struct ValidationError {
    /// Type of validation error
    pub kind: ValidationErrorKind,
    /// Path to the field that failed validation
    pub path: String,
    /// Additional metadata about the field
    pub field_metadata: Option<ValidationFieldMetadata>,
    /// Parent error if this is part of a chain
    pub parent_error: Option<Box<ValidationError>>,
}

impl ValidationError {
    /// Creates a new validation error with metadata
    pub fn new(kind: ValidationErrorKind, path: String, metadata: Option<ValidationFieldMetadata>) -> Self {
        ValidationError {
            kind,
            path,
            field_metadata: metadata,
            parent_error: None,
        }
    }

    /// Creates a new validation error with a parent error
    pub fn with_parent(mut self, parent: ValidationError) -> Self {
        self.parent_error = Some(Box::new(parent));
        self
    }

    /// Gets the severity level of the error
    pub fn severity(&self) -> ValidationSeverity {
        match &self.kind {
            ValidationErrorKind::RuleFailed(_, severity) => *severity,
            ValidationErrorKind::SerializationError(_, _) => ValidationSeverity::Critical,
            ValidationErrorKind::TypeMismatch(_, _) => ValidationSeverity::Error,
            ValidationErrorKind::RegistryError(_, _) => ValidationSeverity::Critical,
            ValidationErrorKind::AggregateError(errors) => {
                errors.iter()
                    .map(|e| e.severity())
                    .max()
                    .unwrap_or(ValidationSeverity::Error)
            }
        }
    }

    /// Returns true if this is a critical error
    pub fn is_critical(&self) -> bool {
        self.severity() == ValidationSeverity::Critical
    }

    /// Returns true if this error has a parent error
    pub fn has_parent(&self) -> bool {
        self.parent_error.is_some()
    }

    /// Gets a detailed error report including the full error chain
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        let mut current = Some(self);
        let mut depth = 0;

        while let Some(error) = current {
            let indent = "  ".repeat(depth);
            report.push_str(&format!("{}At path '{}': {}\n", indent, error.path, error.kind));
            
            if let Some(metadata) = &error.field_metadata {
                report.push_str(&format!("{}Field information:\n", indent));
                report.push_str(&format!("{}  - Name: {}\n", indent, metadata.field_name));
                report.push_str(&format!("{}  - Expected type: {}\n", indent, metadata.expected_type));
                report.push_str(&format!("{}  - Valid values: {}\n", indent, metadata.valid_values));
                report.push_str(&format!("{}  - Required: {}\n", indent, metadata.required));
            }

            current = error.parent_error.as_deref();
            depth += 1;
        }

        report
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] At path '{}': {}", self.severity(), self.path, self.kind)
    }
}

impl Error for ValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.kind)
    }
}

/// Validation result type alias
pub type ValidationResult = Result<(), ValidationError>;

/// Helper functions for creating validation errors
pub mod validation_errors {
    use super::*;

    pub fn rule_failed(msg: &str, path: &str, severity: ValidationSeverity, metadata: Option<ValidationFieldMetadata>) -> ValidationError {
        ValidationError::new(
            ValidationErrorKind::RuleFailed(msg.to_string(), severity),
            path.to_string(),
            metadata
        )
    }

    pub fn type_mismatch(msg: &str, expected_type: &str, path: &str, metadata: Option<ValidationFieldMetadata>) -> ValidationError {
        ValidationError::new(
            ValidationErrorKind::TypeMismatch(msg.to_string(), expected_type.to_string()),
            path.to_string(),
            metadata
        )
    }

    pub fn serialization_error(msg: &str, source: Option<Box<dyn Error + Send + Sync>>, path: &str) -> ValidationError {
        ValidationError::new(
            ValidationErrorKind::SerializationError(msg.to_string(), source),
            path.to_string(),
            None
        )
    }

    pub fn registry_error(msg: &str, source: Option<Box<dyn Error + Send + Sync>>, path: &str) -> ValidationError {
        ValidationError::new(
            ValidationErrorKind::RegistryError(msg.to_string(), source),
            path.to_string(),
            None
        )
    }

    pub fn aggregate_error(errors: Vec<ValidationError>, path: &str) -> ValidationError {
        ValidationError::new(
            ValidationErrorKind::AggregateError(errors),
            path.to_string(),
            None
        )
    }
}

/// Validation rule trait for implementing custom validation rules
pub trait ValidationRule: Send + Sync {
    /// Validates the given value and returns Ok(()) if valid, or ValidationError if invalid
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult;
    /// Returns a description of the validation rule
    fn description(&self) -> String;
}

/// Validation context containing the value being validated and its path
pub struct ValidationContext<'a> {
    pub value: &'a dyn Serialize,
    pub path: String,
}

/// Global validation rule registry for thread-safe rule management
lazy_static! {
    static ref VALIDATION_REGISTRY: RwLock<HashMap<String, Arc<dyn ValidationRule>>> = RwLock::new(HashMap::new());
}

/// Validator struct that holds validation rules and performs validation
pub struct Validator {
    rules: Vec<Arc<dyn ValidationRule>>,
}

impl Validator {
    /// Creates a new validator
    pub fn new() -> Self {
        Validator {
            rules: Vec::new(),
        }
    }

    /// Adds a validation rule to the validator
    pub fn add_rule(&mut self, rule: Arc<dyn ValidationRule>) {
        self.rules.push(rule);
    }

    /// Registers a global validation rule that can be used across threads
    pub fn register_global_rule(name: &str, rule: Arc<dyn ValidationRule>) -> ValidationResult {
        let mut registry = VALIDATION_REGISTRY.write().map_err(|e| ValidationError {
            kind: ValidationErrorKind::RegistryError(format!("Failed to acquire write lock: {}", e)),
            path: String::from("registry"),
        })?;
        registry.insert(name.to_string(), rule);
        Ok(())
    }

    /// Gets a registered global validation rule
    pub fn get_global_rule(name: &str) -> Result<Arc<dyn ValidationRule>, ValidationError> {
        let registry = VALIDATION_REGISTRY.read().map_err(|e| ValidationError {
            kind: ValidationErrorKind::RegistryError(format!("Failed to acquire read lock: {}", e)),
            path: String::from("registry"),
        })?;
        
        registry.get(name).cloned().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::RegistryError(format!("Rule '{}' not found", name)),
            path: String::from("registry"),
        })
    }

    /// Validates a value against all registered rules
    pub fn validate(&self, context: ValidationContext) -> ValidationResult {
        let mut errors = Vec::new();
        
        for rule in &self.rules {
            if let Err(error) = rule.validate(context.value, &context.path) {
                errors.push(error);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationError {
                kind: ValidationErrorKind::AggregateError(errors),
                path: context.path.clone(),
            })
        }
    }
}

/// Common validation rules

/// Metric name validation rule
pub struct MetricNameRule {
    pattern: regex::Regex,
}

impl MetricNameRule {
    pub fn new() -> Self {
        MetricNameRule {
            pattern: regex::Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$").unwrap(),
        }
    }
}

impl ValidationRule for MetricNameRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize metric name: {}", e)),
            path: path.to_string(),
        })?;
            
        let str_value = value.as_str().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Metric name must be a string".to_string()),
            path: path.to_string(),
        })?;
        
        if !self.pattern.is_match(str_value) {
            Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed(
                    "Metric name must start with a letter and contain only alphanumeric characters and underscores (max 64 chars)".to_string()
                ),
                path: path.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn description(&self) -> String {
        "Metric name must start with a letter and contain only alphanumeric characters and underscores (max 64 chars)".to_string()
    }
}

/// Metric value validation rule
pub struct MetricValueRule {
    min: f64,
    max: f64,
}

impl MetricValueRule {
    pub fn new() -> Self {
        MetricValueRule {
            min: f64::MIN,
            max: f64::MAX,
        }
    }

    pub fn with_bounds(min: f64, max: f64) -> Self {
        MetricValueRule { min, max }
    }
}

impl ValidationRule for MetricValueRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize metric value: {}", e)),
            path: path.to_string(),
        })?;
        
        let num_value = value.as_f64().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Metric value must be a number".to_string()),
            path: path.to_string(),
        })?;
        
        if num_value.is_nan() || num_value.is_infinite() {
            return Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed("Metric value must be a finite number".to_string()),
                path: path.to_string(),
            });
        }
        
        if num_value < self.min || num_value > self.max {
            Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed(
                    format!("Metric value must be between {} and {}", self.min, self.max)
                ),
                path: path.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn description(&self) -> String {
        format!("Metric value must be a finite number between {} and {}", self.min, self.max)
    }
}

/// Configuration structure validation rule
pub struct ConfigurationStructureRule {
    required_fields: Vec<String>,
    allowed_types: HashMap<String, Vec<String>>,
    max_metrics: usize,
    max_tag_count: usize,
    max_metric_value: f64,
    min_metric_value: f64,
    max_interval: f64,
    min_interval: f64,
    max_nested_level: usize,
    max_total_tags: usize,
}

impl ConfigurationStructureRule {
    pub fn new() -> Self {
        let mut allowed_types = HashMap::new();
        allowed_types.insert("metrics".to_string(), vec!["object".to_string()]);
        allowed_types.insert("interval".to_string(), vec!["number".to_string()]);
        allowed_types.insert("enabled".to_string(), vec!["boolean".to_string()]);
        allowed_types.insert("description".to_string(), vec!["string".to_string()]);
        allowed_types.insert("tags".to_string(), vec!["object".to_string()]);
        
        ConfigurationStructureRule {
            required_fields: vec!["metrics".to_string(), "interval".to_string()],
            allowed_types,
            max_metrics: 100,
            max_tag_count: 50,
            max_metric_value: 1e9,
            min_metric_value: -1e9,
            max_interval: 3600.0, // 1 hour in seconds
            min_interval: 1.0,    // 1 second minimum interval
            max_nested_level: 5,  // Maximum nesting level for metrics and tags
            max_total_tags: 1000, // Maximum total number of tags across all metrics
        }
    }

    pub fn with_max_metrics(mut self, max_metrics: usize) -> Self {
        self.max_metrics = max_metrics;
        self
    }

    pub fn with_max_tag_count(mut self, max_tag_count: usize) -> Self {
        self.max_tag_count = max_tag_count;
        self
    }

    pub fn with_metric_value_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_metric_value = min;
        self.max_metric_value = max;
        self
    }

    pub fn with_interval_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_interval = min;
        self.max_interval = max;
        self
    }

    pub fn with_max_nested_level(mut self, max_nested_level: usize) -> Self {
        self.max_nested_level = max_nested_level;
        self
    }

    pub fn with_max_total_tags(mut self, max_total_tags: usize) -> Self {
        self.max_total_tags = max_total_tags;
        self
    }
}

impl ValidationRule for ConfigurationStructureRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize configuration: {}", e)),
            path: path.to_string(),
        })?;
        
        let obj = value.as_object().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Configuration must be an object".to_string()),
            path: path.to_string(),
        })?;
        
        // Check required fields
        let missing_fields: Vec<_> = self.required_fields.iter()
            .filter(|field| !obj.contains_key(*field))
            .collect();
            
        if !missing_fields.is_empty() {
            return Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed(
                    format!("Missing required fields: {}", missing_fields.join(", "))
                ),
                path: path.to_string(),
            });
        }
        
        // Validate field types and additional constraints
        let mut errors = Vec::new();
        
        for (field, value) in obj {
            if let Some(allowed_types) = self.allowed_types.get(field) {
                let value_type = match value {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                };
                
                if !allowed_types.contains(&value_type.to_string()) {
                    errors.push(ValidationError {
                        kind: ValidationErrorKind::TypeMismatch(
                            format!("Field '{}' must be one of: {}", field, allowed_types.join(", "))
                        ),
                        path: format!("{}.{}", path, field),
                    });
                }

                // Additional validation for specific fields
                match field.as_str() {
                    "metrics" => {
                        if let serde_json::Value::Object(metrics) = value {
                            if metrics.len() > self.max_metrics {
                                errors.push(ValidationError {
                                    kind: ValidationErrorKind::RuleFailed(
                                        format!("Number of metrics exceeds maximum limit of {}", self.max_metrics)
                                    ),
                                    path: format!("{}.metrics", path),
                                });
                            }

                            // Track total number of tags across all metrics
                            let mut total_tags = 0;
                            
                            // Validate each metric
                            for (metric_name, metric_value) in metrics {
                                // Validate metric name length and pattern
                                if metric_name.len() > 64 {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Metric name '{}' exceeds maximum length of 64 characters", metric_name)
                                        ),
                                        path: format!("{}.metrics.{}", path, metric_name),
                                    });
                                }
                                
                                let name_pattern = regex::Regex::new(r"^[a-zA-Z][a-zA-Z0-9_]*$").unwrap();
                                if !name_pattern.is_match(&metric_name) {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Metric name '{}' must start with a letter and contain only alphanumeric characters and underscores", metric_name)
                                        ),
                                        path: format!("{}.metrics.{}", path, metric_name),
                                    });
                                }
                                
                                // Validate metric value structure and bounds
                                if let serde_json::Value::Object(metric_obj) = metric_value {
                                    if !metric_obj.contains_key("value") {
                                        errors.push(ValidationError {
                                            kind: ValidationErrorKind::RuleFailed(
                                                format!("Metric '{}' must contain a 'value' field", metric_name)
                                            ),
                                            path: format!("{}.metrics.{}", path, metric_name),
                                        });
                                    } else if let Some(value) = metric_obj.get("value") {
                                        if let Some(num_value) = value.as_f64() {
                                            if num_value < self.min_metric_value || num_value > self.max_metric_value {
                                                errors.push(ValidationError {
                                                    kind: ValidationErrorKind::RuleFailed(
                                                        format!("Metric '{}' value must be between {} and {}", 
                                                            metric_name, self.min_metric_value, self.max_metric_value)
                                                    ),
                                                    path: format!("{}.metrics.{}.value", path, metric_name),
                                                });
                                            }
                                        } else {
                                            errors.push(ValidationError {
                                                kind: ValidationErrorKind::TypeMismatch(
                                                    format!("Metric '{}' value must be a number", metric_name)
                                                ),
                                                path: format!("{}.metrics.{}.value", path, metric_name),
                                            });
                                        }
                                    }

                                    // Validate metric tags if present
                                    if let Some(serde_json::Value::Object(metric_tags)) = metric_obj.get("tags") {
                                        if metric_tags.len() > self.max_tag_count {
                                            errors.push(ValidationError {
                                                kind: ValidationErrorKind::RuleFailed(
                                                    format!("Metric '{}' has too many tags (maximum {})", 
                                                        metric_name, self.max_tag_count)
                                                ),
                                                path: format!("{}.metrics.{}.tags", path, metric_name),
                                            });
                                        }

                                        // Update total tag count
                                        total_tags += metric_tags.len();
                                        if total_tags > self.max_total_tags {
                                            errors.push(ValidationError {
                                                kind: ValidationErrorKind::RuleFailed(
                                                    format!("Total number of tags across all metrics exceeds maximum limit of {}", 
                                                        self.max_total_tags)
                                                ),
                                                path: format!("{}.metrics", path),
                                            });
                                        }

                                        // Validate nested level
                                        let mut current_level = 0;
                                        let mut stack = vec![(metric_tags, current_level)];
                                        
                                        while let Some((current_obj, level)) = stack.pop() {
                                            if level > self.max_nested_level {
                                                errors.push(ValidationError {
                                                    kind: ValidationErrorKind::RuleFailed(
                                                        format!("Metric '{}' exceeds maximum nesting level of {}", 
                                                            metric_name, self.max_nested_level)
                                                    ),
                                                    path: format!("{}.metrics.{}.tags", path, metric_name),
                                                });
                                                break;
                                            }
                                            
                                            for (_, value) in current_obj {
                                                if let serde_json::Value::Object(nested_obj) = value {
                                                    stack.push((nested_obj, level + 1));
                                                }
                                            }
                                        }
                                        
                                        for (tag_name, tag_value) in metric_tags {
                                            // Validate tag name
                                            if tag_name.len() > 32 {
                                                errors.push(ValidationError {
                                                    kind: ValidationErrorKind::RuleFailed(
                                                        format!("Tag name '{}' in metric '{}' exceeds maximum length of 32 characters",
                                                            tag_name, metric_name)
                                                    ),
                                                    path: format!("{}.metrics.{}.tags.{}", path, metric_name, tag_name),
                                                });
                                            }
                                            
                                            // Validate tag value
                                            if let Some(str_value) = tag_value.as_str() {
                                                if str_value.len() > 256 {
                                                    errors.push(ValidationError {
                                                        kind: ValidationErrorKind::RuleFailed(
                                                            format!("Tag value for '{}' in metric '{}' exceeds maximum length of 256 characters",
                                                                tag_name, metric_name)
                                                        ),
                                                        path: format!("{}.metrics.{}.tags.{}", path, metric_name, tag_name),
                                                    });
                                                }
                                            } else {
                                                errors.push(ValidationError {
                                                    kind: ValidationErrorKind::TypeMismatch(
                                                        format!("Tag value for '{}' in metric '{}' must be a string",
                                                            tag_name, metric_name)
                                                    ),
                                                    path: format!("{}.metrics.{}.tags.{}", path, metric_name, tag_name),
                                                });
                                            }
                                        }
                                    }
                                } else {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::TypeMismatch(
                                            format!("Metric '{}' must be an object", metric_name)
                                        ),
                                        path: format!("{}.metrics.{}", path, metric_name),
                                    });
                                }
                            }
                        }
                    },
                    "interval" => {
                        if let serde_json::Value::Number(n) = value {
                            if let Some(interval) = n.as_f64() {
                                if interval < self.min_interval || interval > self.max_interval {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Interval must be between {} and {} seconds",
                                                self.min_interval, self.max_interval)
                                        ),
                                        path: format!("{}.interval", path),
                                    });
                                }
                            }
                        }
                    },
                    "tags" => {
                        if let serde_json::Value::Object(tags) = value {
                            for (tag_name, tag_value) in tags {
                                if tag_name.len() > 32 {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::RuleFailed(
                                            format!("Tag name '{}' exceeds maximum length of 32 characters", tag_name)
                                        ),
                                        path: format!("{}.tags.{}", path, tag_name),
                                    });
                                }
                                
                                if let serde_json::Value::String(s) = tag_value {
                                    if s.len() > 256 {
                                        errors.push(ValidationError {
                                            kind: ValidationErrorKind::RuleFailed(
                                                format!("Tag value for '{}' exceeds maximum length of 256 characters", tag_name)
                                            ),
                                            path: format!("{}.tags.{}", path, tag_name),
                                        });
                                    }
                                } else {
                                    errors.push(ValidationError {
                                        kind: ValidationErrorKind::TypeMismatch(
                                            format!("Tag value for '{}' must be a string", tag_name)
                                        ),
                                        path: format!("{}.tags.{}", path, tag_name),
                                    });
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationError {
                kind: ValidationErrorKind::AggregateError(errors),
                path: path.to_string(),
            })
        }
    }

    fn description(&self) -> String {
        format!(
            "Configuration must be an object with required fields: {}. Fields must have correct types.",
            self.required_fields.join(", ")
        )
    }
}

/// Range validation rule for numeric values
pub struct RangeRule<T> {
    min: T,
    max: T,
    field_name: String,
}

impl<T: PartialOrd + std::fmt::Display> RangeRule<T> {
    pub fn new(min: T, max: T, field_name: String) -> Self {
        RangeRule { min, max, field_name }
    }
}

impl<T: PartialOrd + std::fmt::Display + Serialize + Send + Sync + 'static> ValidationRule for RangeRule<T> {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize {}: {}", self.field_name, e)),
            path: path.to_string(),
        })?;
        
        let num_value = value.as_f64().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch(format!("{} must be a number", self.field_name)),
            path: path.to_string(),
        })?;
        
        let min = serde_json::to_value(&self.min).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize min value: {}", e)),
            path: path.to_string(),
        })?.as_f64().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Invalid min value type".to_string()),
            path: path.to_string(),
        })?;
            
        let max = serde_json::to_value(&self.max).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize max value: {}", e)),
            path: path.to_string(),
        })?.as_f64().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch("Invalid max value type".to_string()),
            path: path.to_string(),
        })?;
        
        if num_value < min || num_value > max {
            Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed(
                    format!("{} must be between {} and {}", self.field_name, self.min, self.max)
                ),
                path: path.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn description(&self) -> String {
        format!("{} must be between {} and {}", self.field_name, self.min, self.max)
    }
}

/// Pattern validation rule for string values
pub struct PatternRule {
    pattern: regex::Regex,
    field_name: String,
}

impl PatternRule {
    pub fn new(pattern: regex::Regex, field_name: String) -> Self {
        PatternRule { pattern, field_name }
    }
}

impl ValidationRule for PatternRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize {}: {}", self.field_name, e)),
            path: path.to_string(),
        })?;
            
        let str_value = value.as_str().ok_or_else(|| ValidationError {
            kind: ValidationErrorKind::TypeMismatch(format!("{} must be a string", self.field_name)),
            path: path.to_string(),
        })?;
        
        if !self.pattern.is_match(str_value) {
            Err(ValidationError {
                kind: ValidationErrorKind::RuleFailed(
                    format!("{} does not match the required pattern: {}", 
                        self.field_name, self.pattern.as_str())
                ),
                path: path.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn description(&self) -> String {
        format!("{} must match pattern: {}", self.field_name, self.pattern.as_str())
    }
}

/// Non-empty collection validation rule
pub struct NonEmptyRule {
    field_name: String,
}

impl NonEmptyRule {
    pub fn new(field_name: String) -> Self {
        NonEmptyRule { field_name }
    }
}

impl ValidationRule for NonEmptyRule {
    fn validate(&self, value: &dyn Serialize, path: &str) -> ValidationResult {
        let value = serde_json::to_value(value).map_err(|e| ValidationError {
            kind: ValidationErrorKind::SerializationError(format!("Failed to serialize {}: {}", self.field_name, e)),
            path: path.to_string(),
        })?;
        
        match value {
            serde_json::Value::Array(arr) if arr.is_empty() => {
                Err(ValidationError {
                    kind: ValidationErrorKind::RuleFailed(format!("{} cannot be empty", self.field_name)),
                    path: path.to_string(),
                })
            }
            serde_json::Value::Array(_) => Ok(()),
            serde_json::Value::Object(obj) if obj.is_empty() => {
                Err(ValidationError {
                    kind: ValidationErrorKind::RuleFailed(format!("{} cannot be empty", self.field_name)),
                    path: path.to_string(),
                })
            }
            serde_json::Value::Object(_) => Ok(()),
            serde_json::Value::String(s) if s.is_empty() => {
                Err(ValidationError {
                    kind: ValidationErrorKind::RuleFailed(format!("{} cannot be empty", self.field_name)),
                    path: path.to_string(),
                })
            }
            serde_json::Value::String(_) => Ok(()),
            _ => Err(ValidationError {
                kind: ValidationErrorKind::TypeMismatch(format!("{} must be an array, object, or string", self.field_name)),
                path: path.to_string(),
            }),
        }
    }

    fn description(&self) -> String {
        format!("{} must not be empty", self.field_name)
    }
}
