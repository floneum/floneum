use log::{set_boxed_logger, set_max_level, Metadata, Record};

use crate::plugins::main::imports::log_to_user;

pub struct Logger {
    level: log::Level,
}

impl Logger {
    pub fn new(level: log::Level) -> Self {
        Self { level }
    }

    pub fn register(self) {
        set_max_level(self.level.to_level_filter());
        set_boxed_logger(Box::new(self)).unwrap();
    }
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = record.args();
            #[allow(unused_mut)]
            let mut message = message.to_string();
            #[cfg(debug_assertions)]
            {
                let file = record.file();
                let line = record.line();
                let module_path = record.module_path();
                message.push_str(&format!(
                    " ({}:{} {})",
                    file.unwrap_or(""),
                    line.unwrap_or(0),
                    module_path.unwrap_or("")
                ));
            }
            log_to_user(&message);
        }
    }

    fn flush(&self) {}
}
