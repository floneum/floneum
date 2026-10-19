//! Bounded diagnostic of immutable GGUF host pages after ordinary GPU uploads.
//! Usage: upload_memory_lab MODEL.gguf retain|reclaim [MiB, at most 256]
use fusor::{Device, Dim, QMatrix, VarBuilder};
use fusor_ir::{dtype::Dtype, target::HostBytes};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};

fn phase(name: &str, bytes: usize, tensors: usize) {
    println!(
        "{}",
        serde_json::json!({"phase": name, "bytes": bytes, "tensors": tensors})
    );
    std::thread::sleep(Duration::from_secs(2));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) || !matches!(args[2].as_str(), "retain" | "reclaim") {
        return Err("usage: upload_memory_lab MODEL.gguf retain|reclaim [MiB <= 256]".into());
    }
    let mib: usize = args.get(3).map_or(Ok(256), |value| value.parse())?;
    if !(1..=256).contains(&mib) {
        return Err("the diagnostic is limited to 1..=256 MiB of GPU weights".into());
    }
    let budget = mib * 1024 * 1024;
    let device = Device::gpu_blocking()?;
    let vb = VarBuilder::from_gguf(&args[1])?;
    let mut names = vb.list_all_keys();
    names.sort();
    let mut weights = Vec::new();
    let mut total = 0;
    phase("before_touch", 0, 0);
    for name in names {
        let raw = vb.get_raw(&name)?;
        let Dtype::Q(fmt) = raw.fmt else { continue };
        let [rows, cols] = raw.shape.as_slice() else {
            continue;
        };
        let len = raw.bytes.len();
        // Bound the transient readback allocation as well as total GPU storage.
        if !(1024 * 1024..=16 * 1024 * 1024).contains(&len) || total + len > budget {
            continue;
        }
        let expected = Sha256::digest(raw.bytes.as_ref());
        let source = if args[2] == "reclaim" {
            raw.bytes.shared_host_bytes()
        } else {
            Arc::new(HostBytes::shared(Arc::new(raw.bytes.clone()), 0..len)?)
        };
        let weight = QMatrix::from_shared_bytes(
            device.graph(),
            fmt,
            raw.layout,
            [Dim::Const(*rows), Dim::Const(*cols)],
            source.clone(),
        )?;
        weights.push((name, weight, source, expected));
        total += len;
    }
    if total < budget / 2 {
        return Err("not enough suitably sized tensors for this diagnostic".into());
    }
    phase("host_touched", total, weights.len());
    for (_, weight, _, _) in &weights {
        weight.tensor.upload()?;
    }
    device.try_flush()?;
    phase("gpu_uploaded", total, weights.len());
    for (name, weight, source, expected) in &weights {
        let actual = weight.tensor.as_slice()?;
        assert_eq!(
            Sha256::digest(actual.bytes()),
            *expected,
            "GPU bytes differ: {name}"
        );
        source.with_slice(|bytes| {
            assert_eq!(
                Sha256::digest(bytes),
                *expected,
                "host bytes differ: {name}"
            )
        });
        weight.tensor.upload()?;
    }
    phase("verified", total, weights.len());
    Ok(())
}
