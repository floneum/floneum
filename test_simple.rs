use std::time::Instant;

fn main() {
    println!("Testing flash attention optimization...");
    
    let start = Instant::now();
    
    // Simulate the basic operations that might be causing issues
    let workgroup_size = 256u32;
    let threads_per_output = 64u32;
    let outputs_per_workgroup = workgroup_size / threads_per_output;
    
    println!("Workgroup size: {}", workgroup_size);
    println!("Threads per output: {}", threads_per_output);
    println!("Outputs per workgroup: {}", outputs_per_workgroup);
    
    // Test some basic calculations
    for i in 0..10 {
        let local_output_id = i % outputs_per_workgroup;
        let thread_in_output = i % threads_per_output;
        println!("Thread {}: local_output_id={}, thread_in_output={}", i, local_output_id, thread_in_output);
    }
    
    let elapsed = start.elapsed();
    println!("Test completed in {:?}", elapsed);
}