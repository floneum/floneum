use crate::plugin_search::PluginSearch;
use crate::Color;
use crate::CurrentNodeInfo;
use dioxus::prelude::*;
use dioxus_router::prelude::*;

#[derive(Routable, Clone)]
#[rustfmt::skip]
enum SidebarRoute {
    #[layout(Links)]
        #[route("/")]
        PluginSearch {},
        #[route("/node")]
        CurrentNodeInfo {},
}

pub fn Sidebar(cx: Scope) -> Element {
    render! {
        Router::<SidebarRoute> {}
    }
}

fn Links(cx: Scope) -> Element {
    render! {
        div {
            class: "h-full w-64 {Color::foreground_color()} {Color::text_color()} border-l {Color::outline_color()} top-0 bottom-0 right-0 z-10 fixed overflow-scroll text-center",
            div {
                class: "flex flex-row overflow-x-scroll divide-x border-b {Color::outline_color()}",
                Link {
                    class: "{Color::foreground_hover()} px-3 py-2 text-sm font-medium {Color::outline_color()}",
                    to: SidebarRoute::PluginSearch {},
                    "Plugin Search"
                }
                Link {
                    class: "{Color::foreground_hover()} px-3 py-2 text-sm font-medium {Color::outline_color()}",
                    to: SidebarRoute::CurrentNodeInfo {},
                    "Current Node"
                }
            }
            Outlet::<SidebarRoute> {}
        }
    }
}
