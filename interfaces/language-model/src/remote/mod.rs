use std::sync::OnceLock;

use thiserror::Error;

mod completion;
pub use completion::*;

mod embedding;
pub use embedding::*;

mod chat;
pub use chat::*;

#[derive(Debug, Default)]
struct OpenAICompatibleClient {
    reqwest_client: reqwest::Client,
    model: Option<String>,
    base_url: String,
    api_key: Option<String>,
    resolved_api_key: OnceLock<String>,
    organization_id: Option<String>,
    project_id: Option<String>,
}

impl OpenAICompatibleClient {
    /// Sets the API key for the builder. (defaults to the environment variable `OPENAI_API_KEY`)
    ///
    /// The API key can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/api-keys).
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Set the base URL of the API. (defaults to `https://api.openai.com/v1/`)
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    /// Set the organization ID for the builder.
    ///
    /// The organization ID can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/general).
    pub fn with_organization_id(mut self, organization_id: &str) -> Self {
        self.organization_id = Some(organization_id.to_string());
        self
    }

    /// Set the project ID for the builder.
    ///
    /// The project ID can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/projects).
    pub fn with_project_id(mut self, project_id: &str) -> Self {
        self.project_id = Some(project_id.to_string());
        self
    }

    /// Set the reqwest client for the builder.
    pub fn with_reqwest_client(mut self, client: reqwest::Client) -> Self {
        self.reqwest_client = client;
        self
    }

    /// Resolve the openai API key from the environment variable `OPENAI_API_KEY` or the provided api key.
    pub fn resolve_api_key(&self) -> Result<String, NoAPIKeyError> {
        if let Some(api_key) = self.resolved_api_key.get() {
            return Ok(api_key.clone());
        }

        let open_api_key = match self.api_key.clone() {
            Some(api_key) => api_key,
            None => std::env::var("OPENAI_API_KEY").map_err(|_| NoAPIKeyError)?,
        };

        self.resolved_api_key.set(open_api_key.clone()).unwrap();

        Ok(open_api_key)
    }
}

#[derive(Debug, Error)]
#[error("No API key was provided in the [OpenAICompatibleClient] builder or the environment variable `OPENAI_API_KEY` was not set")]
pub struct NoAPIKeyError;
