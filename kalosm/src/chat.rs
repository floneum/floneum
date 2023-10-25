// use anyhow::Result;
// use kalosm_language::{Model, SyncModel};

// /// The history of a chat session.
// pub struct ChatHistory<Session> {
//     user_marker: String,
//     assistant_marker: String,
//     messages: Vec<String>,
//     session: Session
// }

// impl<Session> ChatHistory<Cache> {
//     /// Creates a new chat history.
//     pub fn new<Model: SyncModel<Session = Session>>(model: &Model, user_marker: String, assistant_marker: String) -> Result<Self> {
//         Ok(Self {
//             user_marker,
//             assistant_marker,
//             messages: Vec::new(),
//             session: model.new_session()?
//         })
//     }

//     /// Adds a message to the history.
//     pub fn add_message(&mut self, message: String, is_user: bool) {
//         let marker = if is_user {
//             &self.user_marker
//         } else {
//             &self.assistant_marker
//         };
//         self.messages.push(format!("{} {}", marker, message));
//     }
// }

// // <s>[INST] <<SYS>>
// // {{ system_prompt }}
// // <</SYS>>

// // {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

// /// A chat session.
// pub struct Chat<'a, M: Model> {
//     model: &'a mut M,
// }

// /// A model that has a chat format.
// pub trait ChatModel {
//     fn user_marker(&self) -> &str;
//     fn assistant_marker(&self) -> &str;
//     fn start_chat(&mut self);
// }
