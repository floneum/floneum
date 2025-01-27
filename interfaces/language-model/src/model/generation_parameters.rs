use std::hash::Hash;
use std::hash::Hasher;

use llm_samplers::configure::SamplerChainBuilder;
use llm_samplers::prelude::*;

/// Parameters to use when generating text.
#[derive(Debug)]
pub struct GenerationParameters {
    pub(crate) temperature: f32,
    pub(crate) tau: f32,
    pub(crate) eta: f32,
    pub(crate) mu: f32,
    pub(crate) top_p: f64,
    pub(crate) top_k: u32,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_penalty_range: u32,
    pub(crate) max_length: u32,
    pub(crate) stop_on: Option<String>,
    pub(crate) seed: Option<u64>,
    sampler: Option<(u64, SamplerChain)>,
}

impl PartialEq for GenerationParameters {
    fn eq(&self, other: &Self) -> bool {
        self.temperature == other.temperature
            && self.eta == other.eta
            && self.tau == other.tau
            && self.mu == other.mu
            && self.top_p == other.top_p
            && self.repetition_penalty == other.repetition_penalty
            && self.repetition_penalty_range == other.repetition_penalty_range
            && self.max_length == other.max_length
            && self.stop_on == other.stop_on
    }
}

impl Clone for GenerationParameters {
    fn clone(&self) -> Self {
        Self {
            temperature: self.temperature,
            eta: self.eta,
            tau: self.tau,
            mu: self.mu,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_range: self.repetition_penalty_range,
            max_length: self.max_length,
            stop_on: self.stop_on.clone(),
            sampler: None,
            seed: None,
        }
    }
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GenerationParameters {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        self.with_sampler(|sampler| sampler.sample(res, logits))
    }

    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &mut Logits,
    ) -> anyhow::Result<Option<TID>> {
        self.with_sampler(|sampler| sampler.sample_token(res, logits))
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.sampler().sampled_token_id()
    }
}

impl GenerationParameters {
    /// Create a new [`GenerationParameters`]
    pub const fn new() -> Self {
        Self {
            temperature: 0.8,
            eta: 0.1,
            tau: 5.,
            mu: 10.,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.3,
            repetition_penalty_range: 64,
            max_length: u32::MAX,
            stop_on: None,
            sampler: None,
            seed: None,
        }
    }

    fn with_sampler<O>(&mut self, with_sampler: impl FnOnce(&mut SamplerChain) -> O) -> O {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.eta.to_le_bytes().hash(&mut hash);
        self.mu.to_le_bytes().hash(&mut hash);
        self.repetition_penalty.to_le_bytes().hash(&mut hash);
        self.repetition_penalty_range.hash(&mut hash);
        self.tau.to_le_bytes().hash(&mut hash);
        self.top_p.to_le_bytes().hash(&mut hash);
        self.temperature.to_le_bytes().hash(&mut hash);
        self.max_length.hash(&mut hash);
        let hash = hash.finish();
        if let Some((old_hash, sampler)) = &mut self.sampler {
            if *old_hash == hash {
                return with_sampler(sampler);
            }
        }
        let mut sampler = self.sampler();
        let output = with_sampler(&mut sampler);
        self.sampler = Some((hash, sampler));
        output
    }

    /// Create a sampler chain from the generation parameters.
    pub fn sampler(&self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            tau,
            eta,
            mu,
            repetition_penalty,
            repetition_penalty_range,
            top_p: _,
            max_length: _,
            stop_on: _,
            ..
        } = self;
        let temperature = *temperature;
        let tau = *tau;
        let eta = *eta;
        let mu = *mu;
        let repetition_penalty = *repetition_penalty;
        let repetition_penalty_range = *repetition_penalty_range;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_static(move || {
                    Box::new(
                        SampleRepetition::default()
                            .penalty(repetition_penalty)
                            .last_n(repetition_penalty_range as usize),
                    )
                }),
            ),
            (
                "freqpresence",
                SamplerSlot::new_static(move || Box::new(SampleFreqPresence::default().last_n(64))),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_static(move || Box::<SampleSeqRepetition>::default()),
            ),
            (
                "temperature",
                SamplerSlot::new_static(move || {
                    Box::new(SampleTemperature::default().temperature(temperature))
                }),
            ),
            (
                "mirostat2",
                SamplerSlot::new_static(move || {
                    Box::new(SampleMirostat2::default().tau(tau).eta(eta).mu(mu))
                }),
            ),
        ])
        .into_chain()
    }

    /// Set the top_p parameter to the generation parameters (only used by the OpenAI API).
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the top_k parameter to the generation parameters (only used by the Anthropic API).
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Get the mirostat2 sampler from the generation parameters.
    pub fn mirostat2_sampler(self) -> SampleMirostat2 {
        SampleMirostat2::default()
            .tau(self.tau)
            .eta(self.eta)
            .mu(self.mu)
    }

    /// Create a sampler chain from the generation parameters without removing any tokens. This can be useful in combination with [`ModelExt::stream_structured_text_with_sampler`] which may pick unlikely tokens.
    pub fn bias_only_sampler(self) -> SamplerChain {
        use llm_samplers::configure::SamplerSlot;
        let GenerationParameters {
            temperature,
            repetition_penalty,
            repetition_penalty_range,
            ..
        } = self;
        SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_static(move || {
                    Box::new(
                        SampleRepetition::default()
                            .penalty(repetition_penalty)
                            .last_n(repetition_penalty_range as usize),
                    )
                }),
            ),
            (
                "freqpresence",
                SamplerSlot::new_static(move || Box::new(SampleFreqPresence::default().last_n(64))),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_static(move || Box::<SampleSeqRepetition>::default()),
            ),
            (
                "temperature",
                SamplerSlot::new_static(move || {
                    Box::new(SampleTemperature::default().temperature(temperature))
                }),
            ),
        ])
        .into_chain()
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the tau to use when generating text.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Set the eta to use when generating text.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Set the mu to use when generating text.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    /// Set the repetition penalty to use when generating text.
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    /// Set the repetition penalty range to use when generating text.
    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.repetition_penalty_range = repetition_penalty_range;
        self
    }

    /// Set the maximum length to use when generating text.
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the string to stop on when generating text.
    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }

    /// Set the seed to use when generating text.
    pub fn with_seed(mut self, seed: impl Into<Option<u64>>) -> Self {
        self.seed = seed.into();
        self
    }

    /// Get the temperature to use when generating text.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the tau to use when generating text.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Get the eta to use when generating text.
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Get the mu to use when generating text.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Get the repetition penalty to use when generating text.
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    /// Get the repetition penalty range to use when generating text.
    pub fn repetition_penalty_range(&self) -> u32 {
        self.repetition_penalty_range
    }

    /// Get the maximum length to use when generating text.
    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    /// Get the string to stop on when generating text.
    pub fn stop_on(&self) -> Option<&str> {
        self.stop_on.as_deref()
    }

    /// Get the seed to use when generating text.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }
}
