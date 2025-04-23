use std::{borrow::Cow, sync::Arc};

use crate::PerformanceQueries;

#[derive(Debug)]
struct DeviceInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    query: Option<Arc<PerformanceQueries>>,
}

#[derive(Clone, Debug)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    pub async fn new() -> Result<Self, crate::Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::SHADER_F16,
                ..Default::default()
            })
            .await?;

        let timing_information = true;
        let query = timing_information.then(|| Arc::new(PerformanceQueries::new(&device, &queue)));

        let device = Self {
            inner: Arc::new(DeviceInner {
                device,
                queue,
                query,
            }),
        };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn({
            let device = device.clone();
            move || loop {
                let Ok(status) = device.wgpu_device().poll(wgpu::PollType::Wait) else {
                    break;
                };
                if status == wgpu::PollStatus::QueueEmpty {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        });

        Ok(device)
    }

    pub(crate) fn create_shader_module<'a>(
        &self,
        source: impl Into<Cow<'a, str>>,
    ) -> wgpu::ShaderModule {
        self.inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(crate) fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub fn query(&self) -> Option<Arc<PerformanceQueries>> {
        self.inner.query.clone()
    }
}
