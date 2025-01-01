use std::fmt::Display;

use minijinja::{context, Environment, ErrorKind};
use minijinja_contrib::pycompat;

use crate::ChatHistoryItem;

#[cfg(test)]
use pretty_assertions::assert_eq;

pub(crate) struct HuggingFaceChatTemplate {
    environment: Environment<'static>,
}

impl HuggingFaceChatTemplate {
    pub(crate) fn create(chat_template: impl Display) -> Result<Self, minijinja::Error> {
        let chat_template = chat_template.to_string();
        let mut environment = Environment::new();

        // enable python compatibility methods because most models are tested with python
        environment.set_unknown_method_callback(pycompat::unknown_method_callback);

        // add the raise_exception function from huggingface templates to the environment
        let raise_exception = |err_text: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                ErrorKind::InvalidOperation,
                format!("The template raised an exception: {}", err_text),
            ))
        };
        environment.add_function("raise_exception", raise_exception);

        // compile the template expression in the environment
        environment.add_template_owned("main", chat_template)?;

        Ok(Self { environment })
    }

    fn run(
        &self,
        bos_token: &str,
        eos_token: &str,
        messages: &[ChatHistoryItem],
        add_generation_prompt: bool,
    ) -> Result<String, minijinja::Error> {
        let ctx = context! { bos_token, eos_token, messages, add_generation_prompt };
        let template = self.environment.get_template("main")?;
        let result = template.render(&ctx)?;
        Ok(result)
    }
}

#[test]
fn test_qwen_chat_template() {
    use crate::MessageType;

    let template = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "Hello, how are you?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::ModelAnswer,
            content: "I'm doing great. How can I help you today?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "I'd like to show off how chat templating works!".to_string(),
        },
    ];

    let result = template
        .run("<|endoftext|>", "<|im_end|>", &inputs, false)
        .unwrap();
    assert_eq!(
        result,
        r#"<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>user
I'd like to show off how chat templating works!<|im_end|>
"#
    );
}

#[test]
fn test_llama_chat_template() {
    use crate::MessageType;

    let template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "Hello, how are you?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::ModelAnswer,
            content: "I'm doing great. How can I help you today?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "I'd like to show off how chat templating works!".to_string(),
        },
    ];

    let result = template
        .run("<|begin_of_text|>", "<|end_of_text|>", &inputs, false)
        .unwrap();

    assert_eq!(
        result,
        r#"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm doing great. How can I help you today?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'd like to show off how chat templating works!<|eot_id|>"#
    )
}

#[test]
fn test_mistral_chat_template() {
    use crate::MessageType;
    let template = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n";

    let template = HuggingFaceChatTemplate::create(template).unwrap();

    let inputs = [
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "Hello, how are you?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::ModelAnswer,
            content: "I'm doing great. How can I help you today?".to_string(),
        },
        ChatHistoryItem {
            role: MessageType::UserMessage,
            content: "I'd like to show off how chat templating works!".to_string(),
        },
    ];

    let result = template.run("<s>", "</s>", &inputs, false).unwrap();
    assert_eq!(
        result,
        r#"<s> [INST] Hello, how are you? [/INST] I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"#
    )
}
