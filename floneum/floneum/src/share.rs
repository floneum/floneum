use crate::{Color, DeserializeApplicationState};
use dioxus::prelude::*;
use std::{fmt::Display, str::FromStr};

use dioxus_std::clipboard::use_clipboard;
use serde::{de::DeserializeOwned, Serialize};

use crate::use_application_state;

pub(crate) fn SaveMenu() -> Element {
    let set_application_state: Coroutine<DeserializeApplicationState> =
        use_coroutine_handle();
    let application = use_application_state();
    let current_application = application.read();
    let current_save_id = &current_application.last_save_id;
    let current_save_string = current_save_id
        .as_ref()
        .map(|id| id.to_string())
        .unwrap_or_default();
    let error: Signal<Option<String>> = use_signal(|| None);
    let current_error = error.read();
    let clipboard = use_clipboard();

    rsx! {
        div {
            class: "flex flex-col {Color::text_color()}",
            div {
                class: "flex flex-row",
                input {
                    class: "border-2 rounded-md p-2 {Color::foreground_color()} {Color::text_color()} {Color::outline_color()}",
                    value: "{current_save_string}",
                    oninput: move |event| {
                        application.write().last_save_id = event.value().parse().ok();
                    },
                }

                button {
                    class: "p-2 {Color::foreground_color()} {Color::text_color()}",
                    onclick: move |_| {
                        to_owned![clipboard];
                        async move {
                            let application = application.read();
                            if let Some(id) = &application.last_save_id {
                                clipboard.set(id.to_string()).unwrap();
                            }
                        }
                    },
                    dioxus_free_icons::Icon {
                        class: "w-4 h-4",
                        icon: dioxus_free_icons::icons::io_icons::IoClipboard,
                    }
                }
            }

            button {
                class: "border-2 rounded-md p-2 {Color::outline_color()}",
                onclick: move |_| {
                    async move {
                        let mut application = application.write();
                        match application.last_save_id {
                            Some(ref id) => {
                                if let Err(err) = id.update(&*application).await {
                                    log::error!("Failed to update save: {}", err);
                                }
                            }
                            None => {
                                match StorageId::new(&*application).await{
                                    Ok(id) => {
                                        application.last_save_id = Some(id);
                                    }
                                    Err(err) => {
                                        *error.write() = Some(err.to_string());
                                    }
                                }
                            }
                        }
                    }
                },
                "Save"
            }

            button {
                class: "border-2 rounded-md p-2 {Color::outline_color()}",
                onclick: move |_| {
                    to_owned![set_application_state];
                    async move {
                        let last_save_id = {
                            application.read().last_save_id.clone()
                        };
                        if let Some(id) = last_save_id {
                            set_application_state.send(DeserializeApplicationState {
                                new_state: id,
                            });
                        }
                    }
                },
                "Load"
            }

            p {
                class: "text-sm opacity-50",
                "Note workflows that are shared is public. Do not store sensitive data. Data will be removed after 30 days of inactivity."
            }

            if let Some(error) = &*current_error {
                p {
                    class: "text-sm text-red-500",
                    "{error}"
                }
            }
        }
    }
}

pub(crate) struct StorageId<T>(String, std::marker::PhantomData<T>);

impl<T> FromStr for StorageId<T> {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(StorageId(
            s.split('/').last().unwrap_or(s).to_string(),
            std::marker::PhantomData,
        ))
    }
}

impl<T> Clone for StorageId<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), std::marker::PhantomData)
    }
}

impl<T> Display for StorageId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: Serialize + DeserializeOwned> StorageId<T> {
    pub async fn new(data: &T) -> anyhow::Result<Self> {
        let client = reqwest::Client::new();
        let res = client
            .post("https://jsonblob.com/api/jsonBlob")
            .json(data)
            .send()
            .await?;

        Ok(res
            .headers()
            .get("Location")
            .ok_or_else(|| anyhow::anyhow!("No location header"))?
            .to_str()?
            .parse()?)
    }

    pub async fn load(&self) -> anyhow::Result<T> {
        let client = reqwest::Client::new();
        let url = format!("https://jsonblob.com/api/jsonBlob/{}", self.0);
        let res = client.get(&url).send().await?;

        let body = res.text().await?;

        Ok(serde_json::from_str(&body)?)
    }

    pub async fn update(&self, data: &T) -> anyhow::Result<()> {
        let client = reqwest::Client::new();
        let url = format!("https://jsonblob.com/api/jsonBlob/{}", self.0);
        client.put(&url).json(data).send().await?;

        Ok(())
    }
}

#[test]
fn save_load() {
    #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Data {
        a: i32,
        b: String,
    }
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let data = Data {
            a: 42,
            b: "Hello, world!".to_string(),
        };
        let id = StorageId::new(&data).await.unwrap();
        let loaded = id.load().await.unwrap();
        assert_eq!(loaded, data);
    });
}
