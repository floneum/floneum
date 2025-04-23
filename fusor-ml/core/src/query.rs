use std::{
    fmt::Display,
    sync::{Arc, atomic::AtomicU32},
    time::Duration,
};

use futures_channel::oneshot;
use wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

// Queries:
// * compute start
// * compute end
const NUM_QUERIES: u64 = 2;
const MAX_QUERIES: u64 = 2048;

#[derive(Debug)]
pub struct PerformanceQueries {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    next_unused_query: AtomicU32,
    generation: AtomicU32,
    get_timestamp_period: f64,
}

#[derive(Debug)]
pub struct QueryResults {
    compute_start_end_timestamps: [u64; 2],
    timestamp_period: f64,
}

impl QueryResults {
    fn new(compute_start_end_timestamps: [u64; 2], timestamp_period: f64) -> Self {
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
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let get_timestamp_period = queue.get_timestamp_period() as f64;
        let size = size_of::<u64>() as u64 * NUM_QUERIES * MAX_QUERIES;
        PerformanceQueries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: (NUM_QUERIES * MAX_QUERIES) as u32,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            next_unused_query: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            get_timestamp_period,
        }
    }

    pub(crate) fn compute_timestamp_writes(self: &Arc<Self>) -> QueryItem {
        let mut index = self
            .next_unused_query
            .fetch_add(2, std::sync::atomic::Ordering::SeqCst);
        let add_generation = if index >= MAX_QUERIES as u32 {
            self.next_unused_query
                .store(0, std::sync::atomic::Ordering::SeqCst);
            index = 0;
            1
        } else {
            0
        };
        let generation = self.generation
            .fetch_add(add_generation, std::sync::atomic::Ordering::SeqCst);

        QueryItem {
            index,
            generation,
            in_query: self.clone(),
        }
    }
}

pub(crate) struct QueryItem {
    index: u32,
    generation: u32,
    in_query: Arc<PerformanceQueries>,
}

impl QueryItem {
    pub(crate) fn compute_pass_timestamp_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        let index = self.index;
        wgpu::ComputePassTimestampWrites {
            query_set: &self.in_query.set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.in_query.set,
            self.index..self.index + NUM_QUERIES as u32,
            &self.in_query.resolve_buffer,
            (self.index as u64 / 2).next_multiple_of(QUERY_RESOLVE_BUFFER_ALIGNMENT),
        );
        encoder.copy_buffer_to_buffer(
            &self.in_query.resolve_buffer,
            self.index as u64 * size_of::<u64>() as u64,
            &self.in_query.destination_buffer,
            (self.index as u64 + 2) * size_of::<u64>() as u64,
            NUM_QUERIES * size_of::<u64>() as u64,
        );
    }

    pub async fn wait_for_results(&self) -> QueryResults {
        assert_eq!(self.generation, self.in_query.generation.load(std::sync::atomic::Ordering::SeqCst));
        let (sender, receiver) = oneshot::channel();
        let sliced = self.in_query.destination_buffer.slice(
            (self.index as u64) * size_of::<u64>() as u64
                ..(self.index as u64 + 2) * size_of::<u64>() as u64,
        );
        sliced.map_async(wgpu::MapMode::Read, move |_| _ = sender.send(()));
        let _ = receiver.await;

        let timestamps: [u64; 2] = {
            let timestamp_view = sliced.get_mapped_range();

            let mut timestamps = [0; 2];
            bytemuck::cast_slice_mut(&mut timestamps).copy_from_slice(&timestamp_view);
            timestamps
        };

        self.in_query.destination_buffer.unmap();

        QueryResults::new(timestamps, self.in_query.get_timestamp_period)
    }
}
