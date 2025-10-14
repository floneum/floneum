use std::{
    borrow::Cow,
    fmt::Debug,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::Arc,
};

use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap};
use wgpu::{BindGroupLayout, BufferUsages, PipelineLayout, ShaderModule};

struct DeviceInner {
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    queue: wgpu::Queue,
    cache: Option<wgpu::PipelineCache>,
    cache_file: Option<PathBuf>,
    bind_group_layout_cache:
        RwLock<LruCache<Vec<wgpu::BindGroupLayoutEntry>, BindGroupLayout, FxBuildHasher>>,
    pipeline_layout_cache: RwLock<LruCache<BindGroupLayout, wgpu::PipelineLayout, FxBuildHasher>>,
    shader_module_cache: RwLock<LruCache<String, wgpu::ShaderModule, FxBuildHasher>>,
    compute_pipeline_cache:
        RwLock<LruCache<(PipelineLayout, ShaderModule), wgpu::ComputePipeline, FxBuildHasher>>,
    // Cache for buffer allocations, keyed by size in bytes
    buffer_allocation_cache: RwLock<FxHashMap<(u64, BufferUsages), Vec<Arc<wgpu::Buffer>>>>,
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
                label: Some("Fusor ML Device"),
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
                    label: Some("Fusor ML Pipeline Cache"),
                    fallback: true,
                })
            };
            (Some(pipeline_cache), Some(cache_path))
        } else {
            (None, None)
        };

        let cache_size = const { NonZeroUsize::new(2048).unwrap() };
        let bind_group_layout_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let pipeline_layout_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let shader_module_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let compute_pipeline_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));

        let device = Self {
            inner: Arc::new(DeviceInner {
                device,
                adapter,
                queue,
                cache,
                cache_file,
                bind_group_layout_cache,
                pipeline_layout_cache,
                shader_module_cache,
                compute_pipeline_cache,
                buffer_allocation_cache: RwLock::new(FxHashMap::default()),
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
        // SAFTEY: All kernels don't access memory outside of bounds and don't have unbounded loops
        unsafe {
            self.inner.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Fusor ML Shader Module"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        }
    }

    pub fn limits(&self) -> wgpu::Limits {
        let mut limits = self.inner.adapter.limits();
        limits.max_subgroup_size = limits.max_subgroup_size.max(64);
        limits.min_subgroup_size = limits.min_subgroup_size.max(4);
        limits
    }

    pub fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
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

    pub(crate) fn bind_group_layout_cache(
        &self,
    ) -> &RwLock<LruCache<Vec<wgpu::BindGroupLayoutEntry>, BindGroupLayout, FxBuildHasher>> {
        &self.inner.bind_group_layout_cache
    }

    pub(crate) fn pipeline_layout_cache(
        &self,
    ) -> &RwLock<LruCache<BindGroupLayout, wgpu::PipelineLayout, FxBuildHasher>> {
        &self.inner.pipeline_layout_cache
    }

    pub(crate) fn shader_module_cache(
        &self,
    ) -> &RwLock<LruCache<String, wgpu::ShaderModule, FxBuildHasher>> {
        &self.inner.shader_module_cache
    }

    pub(crate) fn compute_pipeline_cache(
        &self,
    ) -> &RwLock<LruCache<(PipelineLayout, ShaderModule), wgpu::ComputePipeline, FxBuildHasher>>
    {
        &self.inner.compute_pipeline_cache
    }

    /// Try to get a buffer from the allocation cache. Returns None if no buffer of the requested size is available.
    pub(crate) fn get_cached_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<Arc<wgpu::Buffer>> {
        let mut cache = self.inner.buffer_allocation_cache.write();
        cache.get_mut(&(size, usage))?.pop()
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        // Try to get a buffer from the cache first
        self.get_cached_buffer(size, usage).unwrap_or_else(|| {
            let new_buffer = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tensor Buffer"),
                size,
                usage,
                mapped_at_creation: false,
            });

            let buffer = Arc::new(new_buffer);
            self.inner
                .buffer_allocation_cache
                .write()
                .entry((size, usage))
                .or_default()
                .push(buffer.clone());
            buffer
        })
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        let buffer = self.create_buffer(data.len() as u64, usage);
        self.wgpu_queue().write_buffer(&buffer, 0, data);
        buffer
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init_iter(
        &self,
        data: impl IntoIterator<Item = u8>,
        usage: wgpu::BufferUsages,
        len: u64,
    ) -> Arc<wgpu::Buffer> {
        let mut iter = data.into_iter();
        let buffer = self.create_buffer(len, usage);
        if let Some(len) = NonZeroU64::new(buffer.size()) {
            if let Some(mut write) = self.wgpu_queue().write_buffer_with(&buffer, 0, len) {
                write.iter_mut().zip(&mut iter).for_each(|(a, b)| *a = b);
            }
        }
        buffer
    }
}
