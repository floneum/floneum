mod completion;
pub use completion::*;

mod embedding;
pub use embedding::*;

mod chat;
pub use chat::*;


#[derive(Debug, Default)]
struct OpenAICompatibleClient {
    model: Option<String>,
    base_url: String,
    api_key: Option<String>,
    organization_id: Option<String>,
    project_id: Option<String>,
}


impl OpenAICompatibleClient {
    /// Sets the API key for the builder. (defaults to the environment variable `OPENAI_API_KEY`)
    /// 
    /// The API key can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/api-keys).
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.config.api_key = Some(api_key.to_string());
        self
    }

    /// Set the base URL of the API. (defaults to `https://api.openai.com/v1/`)
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.config.base_url = base_url.to_string();
        self
    }

    /// Set the organization ID for the builder.
    /// 
    /// The organization ID can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/general).
    pub fn with_organization_id(mut self, organization_id: &str) -> Self {
        self.config.organization_id = Some(organization_id.to_string());
        self
    }

    /// Set the project ID for the builder.
    /// 
    /// The project ID can be accessed from the OpenAI dashboard [here](https://platform.openai.com/settings/organization/projects).
    pub fn with_project_id(mut self, project_id: &str) -> Self {
        self.config.project_id = Some(project_id.to_string());
        self
    }
}