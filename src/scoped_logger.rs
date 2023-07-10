use std::sync::Mutex;

use log::{Metadata, Record};
use once_cell::sync::Lazy;
use slab::Slab;

static LOGGERS: Lazy<Mutex<Slab<Box<dyn log::Log>>>> = Lazy::new(|| Mutex::new(Slab::new()));

static LOGGER_STACK: Lazy<Mutex<Vec<usize>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub struct ScopedLoggerLevel {
    id: usize,
}

impl Drop for ScopedLoggerLevel {
    fn drop(&mut self) {
        let mut stack = LOGGER_STACK.lock().unwrap();
        stack.retain(|id| *id != self.id);
        if let Some(id) = stack.last().copied() {
            LOGGERS.lock().unwrap().remove(id);
        }
    }
}

pub fn add_logger(logger: impl log::Log + 'static) -> ScopedLoggerLevel {
    let id = LOGGERS.lock().unwrap().insert(Box::new(logger));
    LOGGER_STACK.lock().unwrap().push(id);
    ScopedLoggerLevel { id }
}

pub struct ScopedLogger;

impl ScopedLogger {
    fn with_last<O>(f: impl FnOnce(&dyn log::Log) -> O) -> Option<O> {
        let last_id = LOGGER_STACK.lock().unwrap().last().copied();
        if let Some(id) = last_id {
            let loggers = LOGGERS.lock().unwrap();
            let logger = loggers.get(id).unwrap();
            Some(f(&**logger))
        } else {
            None
        }
    }
}

impl log::Log for ScopedLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        ScopedLogger::with_last(|logger| logger.enabled(metadata)).unwrap_or_default()
    }

    fn log(&self, record: &Record) {
        ScopedLogger::with_last(|logger| logger.log(record));
    }

    fn flush(&self) {}
}
