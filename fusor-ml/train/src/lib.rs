use fusor::{
    Device, Tensor as RawTensor,
    autograd::{Gradients, Tensor},
};

#[derive(Clone, Copy, Debug)]
pub struct AdamWSettings {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    adam_eps: f32,
    weight_decay: f32,
}

pub trait AdamWModel: Sized {
    type State;

    fn adamw_state(device: &Device, model: &Self) -> Self::State;

    fn adamw_step(
        self,
        state: &mut Self::State,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> Self;
}

pub struct AdamW<M: AdamWModel> {
    step: usize,
    settings: AdamWSettings,
    state: M::State,
}

pub struct AdamMoments<const R: usize> {
    m: RawTensor<R, f32>,
    v: RawTensor<R, f32>,
}

impl AdamWSettings {
    pub const fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        adam_eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            adam_eps,
            weight_decay,
        }
    }
}

impl<M: AdamWModel> AdamW<M> {
    pub fn new(device: &Device, model: &M, settings: AdamWSettings) -> Self {
        Self {
            step: 0,
            settings,
            state: M::adamw_state(device, model),
        }
    }

    pub fn step(&mut self, model: M, gradients: &Gradients) -> M {
        self.step += 1;
        model.adamw_step(&mut self.state, gradients, self.step, self.settings)
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.settings.learning_rate = learning_rate;
    }

    pub fn settings(&self) -> AdamWSettings {
        self.settings
    }
}

impl<const R: usize> AdamMoments<R> {
    pub fn zeros_like(device: &Device, parameter: &Tensor<R>) -> Self {
        let shape = parameter.shape();
        Self {
            m: RawTensor::zeros(device, shape),
            v: RawTensor::zeros(device, shape),
        }
    }
}

pub fn adamw_update<const R: usize>(
    parameter: &Tensor<R>,
    moments: &mut AdamMoments<R>,
    gradients: &Gradients,
    step: usize,
    settings: AdamWSettings,
) -> Tensor<R> {
    let gradient = gradients.get(parameter).unwrap();
    let next_m = ((moments.m.clone() * settings.beta1)
        + (gradient.clone() * (1.0 - settings.beta1)))
        .to_concrete();
    let next_v = ((moments.v.clone() * settings.beta2)
        + (gradient.sqr().to_concrete() * (1.0 - settings.beta2)))
        .to_concrete();

    let bias_correction1 = 1.0 - settings.beta1.powi(step as i32);
    let bias_correction2 = 1.0 - settings.beta2.powi(step as i32);
    let m_hat = next_m.clone().div_scalar(bias_correction1).to_concrete();
    let v_hat = next_v.clone().div_scalar(bias_correction2).to_concrete();
    let adam_update =
        (m_hat / (v_hat.add_scalar(settings.adam_eps).sqrt().to_concrete())).to_concrete();
    let weight_decay = (parameter.raw().clone() * settings.weight_decay).to_concrete();
    let next_parameter = (parameter.raw().clone()
        - ((adam_update + weight_decay).to_concrete() * settings.learning_rate))
        .to_concrete();

    moments.m = next_m;
    moments.v = next_v;
    Tensor::from_raw(&parameter.graph(), next_parameter)
}
