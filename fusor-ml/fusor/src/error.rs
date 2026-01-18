//! Error types for the fusor crate

/// Error type for fusor operations.
#[derive(Debug)]
pub enum Error {
    /// GPU device error from fusor-core.
    Gpu(fusor_core::Error),
    /// Device mismatch error - tensors are on different devices.
    DeviceMismatch {
        /// Description of what operation failed.
        operation: &'static str,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Gpu(e) => write!(f, "GPU error: {}", e),
            Error::DeviceMismatch { operation } => {
                write!(
                    f,
                    "Device mismatch in {}: tensors must be on the same device",
                    operation
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Gpu(e) => Some(e),
            Error::DeviceMismatch { .. } => None,
        }
    }
}

impl From<fusor_core::Error> for Error {
    fn from(e: fusor_core::Error) -> Self {
        Error::Gpu(e)
    }
}
