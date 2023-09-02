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
    let script = r##"
const BORDER_SIZE = 4;
const panel = document.getElementById("left_panel");

let m_pos;
function resize(e){
    const dx = m_pos - e.x;
    m_pos = e.x;
    panel.style.width = (parseInt(getComputedStyle(panel, '').width) + dx) + "px";
}

panel.addEventListener("mousedown", function(e){
    if (e.offsetX < BORDER_SIZE) {
    m_pos = e.x;
    document.addEventListener("mousemove", resize, false);
    }
}, false);

document.addEventListener("mouseup", function(){
    document.removeEventListener("mousemove", resize, false);
}, false);
"##;
    render! {
        div {
            id: "left_panel",
            class: "h-full w-64 {Color::foreground_color()} {Color::text_color()} border-l-4 {Color::outline_color()} top-0 bottom-0 right-0 z-10 fixed overflow-scroll text-center",
            div {
                class: "flex flex-row overflow-x-scroll divide-x border-b {Color::outline_color()}",
                Link {
                    class: "{Color::foreground_hover()} {Color::outline_color()} px-3 py-2 text-sm font-medium w-full",
                    to: SidebarRoute::PluginSearch {},
                    "Plugin Search"
                }
                Link {
                    class: "{Color::foreground_hover()} {Color::outline_color()} px-3 py-2 text-sm font-medium w-full",
                    to: SidebarRoute::CurrentNodeInfo {},
                    "Current Node"
                }
            }
            Outlet::<SidebarRoute> {}
        }
        script {
            dangerous_inner_html: script
        }
    }
}
