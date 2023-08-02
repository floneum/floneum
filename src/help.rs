use dioxus::prelude::*;

#[inline_props]
pub fn Help(cx: Scope, help_text: String) -> Element {
    let help_visible = use_state(cx, || false);

    render! {
        if **help_visible {
            rsx! {
                "{help_text}"
            }
        }
        else {
            rsx! {
                button {
                    onclick: move |_| {
                        help_visible.set(true);
                    },
                    "Help"
                }
            }
        }
    }
}
