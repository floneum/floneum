// Icons from https://ionic.io/ionicons

use dioxus::prelude::*;

#[component]
pub fn IoClipboard(
    #[props(extends = GlobalAttributes)] attrs: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! {
        svg {
            xmlns: "http://www.w3.org/2000/svg",
            "viewBox": "0 0 512 512",
            ..attrs,
            path {
                d: "M336 64h32a48 48 0 0148 48v320a48 48 0 01-48 48H144a48 48 0 01-48-48V112a48 48 0 0148-48h32",
                "stroke-linejoin": "round",
                stroke: "currentColor",
                fill: "none",
                "stroke-width": "32"
            }
            rect {
                rx: "26.13",
                ry: "26.13",
                "stroke-linejoin": "round",
                "stroke-width": "32",
                width: "160",
                y: "32",
                stroke: "currentColor",
                x: "176",
                fill: "none",
                height: "64"
            }
        }
    }
}

#[component]
pub fn IoTrashOutline(
    #[props(extends = GlobalAttributes)] attrs: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! {
        svg {
            "viewBox": "0 0 512 512",
            xmlns: "http://www.w3.org/2000/svg",
            ..attrs,
            path {
                "stroke-linejoin": "round",
                "stroke-linecap": "round",
                fill: "none",
                d: "M112 112l20 320c.95 18.49 14.4 32 32 32h184c17.67 0 30.87-13.51 32-32l20-320",
                stroke: "currentColor",
                "stroke-width": "32"
            }
            path {
                "stroke-width": "32",
                stroke: "currentColor",
                "stroke-miterlimit": "10",
                d: "M80 112h352",
                "stroke-linecap": "round"
            }
            path {
                d: "M192 112V72h0a23.93 23.93 0 0124-24h80a23.93 23.93 0 0124 24h0v40M256 176v224M184 176l8 224M328 176l-8 224",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "stroke-width": "32",
                fill: "none",
                stroke: "currentColor"
            }
        }
    }
}
