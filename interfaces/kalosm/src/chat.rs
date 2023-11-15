// use anyhow::Result;
// use kalosm_language::{Model, SyncModel};

// enum ChatState {
//     SystemPrompt,
//     UserMessage,
//     ModelAnswer,
// }

// /// The history of a chat session.
// struct ChatSession<Session> {
//     user_marker: String,
//     assistant_marker: String,
//     messages: String,
//     session: Session,
//     state: ChatState,
// }

// impl<Session> ChatSession<Session> {
//     /// Creates a new chat history.
//     pub fn new<Model: SyncModel<Session = Session>>(
//         model: &Model,
//         system_prompt_marker: String,
//         user_marker: String,
//         assistant_marker: String,
//         system_prompt: String,
//     ) -> Result<Self> {
//         Ok(Self {
//             user_marker,
//             assistant_marker,
//             messages: system_prompt_marker + &system_prompt,
//             session: model.new_session()?,
//             state: ChatState::SystemPrompt,
//         })
//     }

//     /// Adds a message to the history.
//     pub fn add_message<Model: SyncModel<Session = Session>>(&mut self, message: String, is_user: bool, model: &mut Model) -> Result<()> {
//         let marker = if is_user {
//             &self.user_marker
//         } else {
//             &self.assistant_marker
//         };
//         let new_text = format!("{}{}\n", marker, message);
//         model.feed_text(&mut self.session, &new_text)?;
//         self.messages.push_str(&new_text);
//         Ok(())
//     }

//     /// Returns the history as a string.
//     pub fn prompt(&self) -> &str {
//         &self.messages
//     }
// }

// // <s>[INST] <<SYS>>
// // {{ system_prompt }}
// // <</SYS>>

// // {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

// /// A chat session.
// pub struct Chat{

// }

// impl Chat {
//     /// Creates a new chat session.
//     pub fn new<Model: ChatModel>(model: &Model, system_prompt: String) -> Result<Self> {
//         let system_prompt_marker = model.system_prompt_marker().to_string();
//         let user_marker = model.user_marker().to_string();
//         let assistant_marker = model.assistant_marker().to_string();
//         model.run_sync_raw(move ||{
//             Box::pin(async move {

//             })
//         });
//         Ok(Self {  })
//     }

//     /// Adds a message to the history.
//     pub fn add_message(&mut self, message: String, is_user: bool, model: &mut Model) -> Result<()> {
//         self.session.add_message(message, is_user, model)
//     }
// }

// /// A model that has a chat format.
// pub trait ChatModel: Model {
//     fn user_marker(&self) -> &str;
//     fn assistant_marker(&self) -> &str;
//     fn system_prompt_marker(&self) -> &str;
// }
