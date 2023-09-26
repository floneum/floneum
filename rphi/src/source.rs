pub struct PhiSource {
    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    pub(crate) model_id: String,
    pub(crate) revision: String,
}

impl PhiSource {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            revision: "main".to_string(),
        }
    }

    pub fn with_revision(mut self, revision: String) -> Self {
        self.revision = revision;
        self
    }
}

impl Default for PhiSource {
    fn default() -> Self {
        let default_model = "microsoft/phi-1_5".to_string();
        let default_revision = "refs/pr/18".to_string();
        Self {
            model_id: default_model,
            revision: default_revision,
        }
    }
}
