#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_void};

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));


