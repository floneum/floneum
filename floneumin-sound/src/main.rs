use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{LocalRb, Rb, SharedRb};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperToken};

use std::io::Write;
use std::time::{Duration, Instant};
use std::{cmp, thread};

const LATENCY_MS: f32 = 5000.0;
const NUM_ITERS: usize = 2;
const NUM_ITERS_SAVED: usize = 2;

// TODO: JPB: Add clean way to exit besides ctrl+C (which sometimes doesn't work)
// TODO: JPB: Make sure this works with other LATENCY_MS, NUM_ITERS, and NUM_ITERS_SAVED
// TODO: JPB: I think there is an issue where it doesn't compute fast enough and so it loses data

pub fn run_example() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();

    // Default devices.
    let input_device = host
        .default_input_device()
        .expect("failed to get default input device");
    //let output_device = host
    //    .default_output_device()
    //    .expect("failed to get default output device");
    println!("Using default input device: \"{}\"", input_device.name()?);
    //println!("Using default output device: \"{}\"", output_device.name()?);

    // Top level variables
    let config: cpal::StreamConfig = input_device.default_input_config()?.into();
    let latency_frames = (LATENCY_MS / 1_000.0) * config.sample_rate.0 as f32;
    let latency_samples = latency_frames as usize * config.channels as usize;
    let sampling_freq = config.sample_rate.0 as f32 / 2.0; // TODO: JPB: Divide by 2 because of stereo to mono

    // The buffer to share samples
    let ring = SharedRb::new(latency_samples * 2);
    let (mut producer, mut consumer) = ring.split();

    // Setup microphone callback
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut output_fell_behind = false;
        for &sample in data {
            if producer.push(sample).is_err() {
                output_fell_behind = true;
            }
        }
        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    //let output_data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
    //    let mut input_fell_behind = None;
    //    for sample in data {
    //        *sample = match consumer.pop() {
    //            Some(s) => s,
    //            None => {
    //                input_fell_behind = Some("");
    //                0.0
    //            }
    //        };
    //    }
    //    if let Some(err) = input_fell_behind {
    //        eprintln!(
    //            "input stream fell behind: {:?}: try increasing latency",
    //            err
    //        );
    //    }
    //};

    // Setup whisper
    let arg1 = std::env::args()
        .nth(1)
        .expect("First argument should be path to Whisper model");
    let whisper_path = Path::new(&arg1);
    if !whisper_path.exists() && !whisper_path.is_file() {
        panic!("expected a whisper directory")
    }
    let ctx = WhisperContext::new(&whisper_path.to_string_lossy()).expect("failed to open model");
    let mut state = ctx.create_state().expect("failed to create key");

    // Variables used across loop iterations
    let mut iter_samples = LocalRb::new(latency_samples * NUM_ITERS * 2);
    let mut iter_num_samples = LocalRb::new(NUM_ITERS);
    let mut iter_tokens = LocalRb::new(NUM_ITERS_SAVED);
    for _ in 0..NUM_ITERS {
        iter_num_samples
            .push(0)
            .expect("Error initailizing iter_num_samples");
    }

    // Build streams.
    println!(
        "Attempting to build both streams with f32 samples and `{:?}`.",
        config
    );
    println!("Setup input stream");
    let input_stream = input_device.build_input_stream(&config, input_data_fn, err_fn, None)?;
    //println!("Setup output stream");
    //let output_stream = output_device.build_output_stream(&config, output_data_fn, err_fn, None)?;
    println!("Successfully built streams.");

    // Play the streams.
    println!(
        "Starting the input and output streams with `{}` milliseconds of latency.",
        LATENCY_MS
    );
    input_stream.play()?;
    //output_stream.play()?;

    // Remove the initial samples
    consumer.pop_iter().count();
    let mut start_time = Instant::now();

    // Main loop
    // TODO: JPB: Make this it's own function (And the lines above it)
    let mut num_chars_to_delete = 0;
    let mut loop_num = 0;
    let mut words = "".to_owned();
    loop {
        loop_num += 1;

        // Only run every LATENCY_MS
        let duration = start_time.elapsed();
        let latency = Duration::from_millis(LATENCY_MS as u64);
        if duration < latency {
            let sleep_time = latency - duration;
            thread::sleep(sleep_time);
        } else {
            panic!("Classification got behind. It took to long. Try using a smaller model and/or more threads");
        }
        start_time = Instant::now();

        // Collect the samples
        let samples: Vec<_> = consumer.pop_iter().collect();
        let samples = whisper_rs::convert_stereo_to_mono_audio(&samples).unwrap();
        //let samples = make_audio_louder(&samples, 1.0);
        let num_samples_to_delete = iter_num_samples
            .push_overwrite(samples.len())
            .expect("Error num samples to delete is off");
        for _ in 0..num_samples_to_delete {
            iter_samples.pop();
        }
        iter_samples.push_iter(&mut samples.into_iter());
        let (head, tail) = iter_samples.as_slices();
        let current_samples = [head, tail].concat();

        // Get tokens to be deleted
        if loop_num > 1 {
            let num_tokens = state.full_n_tokens(0)?;
            let token_time_end = state.full_get_segment_t1(0)?;
            let token_time_per_ms =
                token_time_end as f32 / (LATENCY_MS * cmp::min(loop_num, NUM_ITERS) as f32); // token times are not a value in ms, they're 150 per second
            let ms_per_token_time = 1.0 / token_time_per_ms;

            let mut tokens_saved = vec![];
            // Skip beginning and end token
            for i in 1..num_tokens - 1 {
                let token = state.full_get_token_data(0, i)?;
                let token_t0_ms = token.t0 as f32 * ms_per_token_time;
                let ms_to_delete = num_samples_to_delete as f32 / (sampling_freq / 1000.0);

                // Save tokens for whisper context
                if (loop_num > NUM_ITERS) && token_t0_ms < ms_to_delete {
                    tokens_saved.push(token.id);
                }
            }
            num_chars_to_delete = words.chars().count();
            if loop_num > NUM_ITERS {
                num_chars_to_delete -= tokens_saved
                    .iter()
                    .map(|x| ctx.token_to_str(*x).expect("Error"))
                    .collect::<String>()
                    .chars()
                    .count();
            }
            iter_tokens.push_overwrite(tokens_saved);
            //println!();
            //println!(
            //    "TOKENS_SAVED : {}",
            //    tokens_saved
            //        .iter()
            //        .map(|x| ctx.token_to_str(*x).unwrap())
            //        .collect::<Vec<_>>()
            //        .join("")
            //);
            //println!(
            //    "CHARS_DELETED: {}",
            //    words[words.len() - num_chars_to_delete..].to_owned()
            //);
            //println!(
            //    "ITER_TOKENS  : {}",
            //    iter_tokens
            //        .iter()
            //        .flatten()
            //        .map(|x| ctx.token_to_str(*x).unwrap())
            //        .collect::<Vec<_>>()
            //        .join("")
            //);
            //println!("WORDS        : {}", words);
            //println!("NUM_CHARS    : {} {}", words.len(), num_chars_to_delete);
        }

        // Make the model params
        let (head, tail) = iter_tokens.as_slices();
        let tokens = [head, tail]
            .concat()
            .into_iter()
            .flatten()
            .collect::<Vec<WhisperToken>>();
        let mut params = gen_whisper_params();
        params.set_tokens(&tokens);

        // Run the model
        state
            .full(params, &current_samples)
            .expect("failed to convert samples");

        // Update the words on screen
        if num_chars_to_delete != 0 {
            // TODO: JPB: Potentially unneeded if statement
            print!(
                "\x1B[{}D{}\x1B[{}D",
                num_chars_to_delete,
                " ".repeat(num_chars_to_delete),
                num_chars_to_delete
            );
        }
        let num_tokens = state.full_n_tokens(0)?;
        words = (1..num_tokens - 1)
            .map(|i| state.full_get_token_text(0, i).expect("Error"))
            .collect::<String>();
        print!("{}", words);
        std::io::stdout().flush().unwrap();
    }
}

fn gen_whisper_params<'a>() -> FullParams<'a, 'a> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_language(Some("en"));
    params.set_token_timestamps(true);
    // params.set_duration_ms(LATENCY_MS as i32);
    // params.set_no_context(true);
    // params.set_n_threads(16);

    //params.set_no_speech_thold(0.3);
    //params.set_split_on_word(true);

    // This impacts token times, don't use
    //params.set_single_segment(true);

    params
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}

fn main() -> Result<(), anyhow::Error> {
    run_example()
}
