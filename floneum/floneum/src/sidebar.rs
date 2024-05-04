use crate::plugin_search::PluginSearch;
// use crate::share::SaveMenu;
use crate::CurrentNodeInfo;
use dioxus::prelude::*;

#[derive(Routable, Clone)]
#[rustfmt::skip]
enum SidebarRoute {
    #[layout(Links)]
        #[route("/")]
        PluginSearch {},
        #[route("/node")]
        CurrentNodeInfo {},
        // #[route("/save")]
        // SaveMenu {}
}

pub fn Sidebar() -> Element {
    rsx! {
        Router::<SidebarRoute> {}
    }
}

fn Links() -> Element {
    let script = r#"
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
"#;
    rsx! {
        div {
            id: "left_panel",
            class: "h-full w-64 cursor-ew-resize select-none border-l-4 border-black top-0 bottom-0 right-0 z-10 fixed overflow-scroll text-center",
            div { class: "flex flex-row overflow-x-scroll divide-x divide-black border-b border-black",
                Link {
                    class: "px-3 py-2 text-sm font-medium w-full",
                    to: SidebarRoute::PluginSearch {},
                    "Plugin Search"
                }
                Link {
                    class: "px-3 py-2 text-sm font-medium w-full",
                    to: SidebarRoute::CurrentNodeInfo {},
                    "Current Node"
                }
            }
            Outlet::<SidebarRoute> {}
        }
        script { dangerous_inner_html: script }
    }
}
