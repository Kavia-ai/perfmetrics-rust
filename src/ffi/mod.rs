#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_void};

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[no_mangle]
pub extern "C" fn performance_metrics_create() -> *mut c_void {
    // TODO: Implement actual creation logic
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn performance_metrics_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        // TODO: Implement cleanup logic
        unsafe {
            drop(Box::from_raw(handle as *mut c_void));
        }
    }
}

#[no_mangle]
pub extern "C" fn performance_metrics_get_version() -> *const c_char {
    // Return version string
    "1.0.0\0".as_ptr() as *const c_char
}