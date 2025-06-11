use std::{borrow::Cow, fmt::Debug, path::PathBuf, sync::Arc};

struct DeviceInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cache: Option<wgpu::PipelineCache>,
    cache_file: Option<PathBuf>,
}

impl Debug for DeviceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceInner")
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish()
    }
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
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16,
                ..Default::default()
            })
            .await?;

        use wgpu::PipelineCacheDescriptor;
        let filename = wgpu::util::pipeline_cache_key(&adapter.get_info());
        let (cache, cache_file) = if let Some(filename) = filename {
            let cache_dir: PathBuf = PathBuf::from(".fusor").join("pipeline_cache");
            let cache_path = cache_dir.join(&filename);
            let cache_data = std::fs::read(&cache_path).ok();
            let pipeline_cache = unsafe {
                device.create_pipeline_cache(&PipelineCacheDescriptor {
                    data: cache_data.as_deref(),
                    label: None,
                    fallback: true,
                })
            };
            (Some(pipeline_cache), Some(cache_path))
        } else {
            (None, None)
        };

        let device = Self {
            inner: Arc::new(DeviceInner {
                device,
                queue,
                cache,
                cache_file,
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
                    std::thread::sleep(std::time::Duration::from_nanos(10));
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

    pub(crate) fn wgpu_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.inner.cache.as_ref()
    }

    pub(crate) fn wgpu_cache_file(&self) -> Option<&PathBuf> {
        self.inner.cache_file.as_ref()
    }
}
