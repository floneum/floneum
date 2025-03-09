use std::{fmt::Display, sync::atomic::AtomicU32, time::Duration};

use futures_channel::oneshot;

// Queries:
// * compute start
// * compute end
const NUM_QUERIES: u64 = 2;

pub struct PerformanceQueries {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    query_count: u64,
    next_unused_query: AtomicU32,
    get_timestamp_period: f64,
}

#[derive(Debug)]
pub struct QueryResults {
    compute_start_end_timestamps: [u64; 2],
    timestamp_period: f64,
}

impl QueryResults {
    fn new(timestamps: Vec<u64>, timestamp_period: f64) -> Self {
        let compute_start_end_timestamps = timestamps.try_into().unwrap();

        QueryResults {
            compute_start_end_timestamps,
            timestamp_period,
        }
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(
            (self.compute_start_end_timestamps[1]
                .saturating_sub(self.compute_start_end_timestamps[0]) as f64
                * self.timestamp_period) as _,
        )
    }
}

impl Display for QueryResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Elapsed time compute pass: {:?}", self.elapsed())
    }
}

impl PerformanceQueries {
    pub fn new(device: &crate::Device) -> Self {
        let get_timestamp_period = device.wgpu_queue().get_timestamp_period() as f64;
        let device = device.wgpu_device();
        PerformanceQueries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: NUM_QUERIES as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: size_of::<u64>() as u64 * NUM_QUERIES,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: size_of::<u64>() as u64 * NUM_QUERIES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            query_count: NUM_QUERIES,
            next_unused_query: AtomicU32::new(0),
            get_timestamp_period,
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            0..self
                .next_unused_query
                .load(std::sync::atomic::Ordering::SeqCst),
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub async fn wait_for_results(&self) -> QueryResults {
        let (sender, receiver) = oneshot::channel();
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_| _ = sender.send(()));
        let _ = receiver.await;

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(..(size_of::<u64>() as wgpu::BufferAddress * self.query_count))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        QueryResults::new(timestamps, self.get_timestamp_period)
    }

    pub fn compute_timestamp_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        let next_unused_query = self
            .next_unused_query
            .fetch_add(2, std::sync::atomic::Ordering::SeqCst);

        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(next_unused_query),
            end_of_pass_write_index: Some(next_unused_query + 1),
        }
    }
}
