use crate::plugin_search::PluginSearch;
use dioxus::prelude::*;
use dioxus_router::prelude::*;
use crate::CurrentNodeInfo;

#[derive(Routable, Clone)]
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
            class: "h-full w-64 bg-gray-800 top-0 bottom-0 right-0 z-10 fixed overflow-scroll",
            div {
                class: "flex flex-row overflow-x-scroll",
                Link {
                    class: "text-white hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium",
                    to: SidebarRoute::PluginSearch {},
                    "Plugin Search"
                }
                Link {
                    class: "text-white hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium",
                    to: SidebarRoute::CurrentNodeInfo {},
                    "Current Node"
                }
            }
            Outlet::<SidebarRoute> {}
        }
    }
}
