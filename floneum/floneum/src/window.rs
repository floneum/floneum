use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use dioxus::desktop::use_wry_event_handler;
use dioxus::desktop::{tao::window::Icon, WindowBuilder};
use dioxus::prelude::*;
use muda::accelerator::Accelerator;
use muda::{Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};
use serde::Serialize;

use crate::{ApplicationState, SAVE_NAME};

pub(crate) fn make_config() -> anyhow::Result<dioxus::desktop::Config> {
    // Add a bunch of built-in menu items
    let main_menu = Menu::new();
    let edit_menu = Submenu::new("Edit", true);
    let window_menu = Submenu::new("Window", true);
    let application_menu = Submenu::new("Floneum", true);
    let examples_menu = Submenu::new("Examples", true);

    edit_menu.append_items(&[
        &PredefinedMenuItem::undo(None),
        &PredefinedMenuItem::redo(None),
        &PredefinedMenuItem::separator(),
        &PredefinedMenuItem::cut(None),
        &PredefinedMenuItem::copy(None),
        &PredefinedMenuItem::paste(None),
        &PredefinedMenuItem::select_all(None),
    ])?;

    window_menu.append_items(&[
        &PredefinedMenuItem::quit(None),
        &PredefinedMenuItem::minimize(None),
        &PredefinedMenuItem::separator(),
        &PredefinedMenuItem::show_all(None),
        &PredefinedMenuItem::fullscreen(None),
        &PredefinedMenuItem::separator(),
        &PredefinedMenuItem::close_window(None),
    ])?;

    application_menu.append_items(&[
        &SavePredefinedMenuItem::item(),
        &SaveAsPredefinedMenuItem::item(),
        &OpenPredefinedMenuItem::item(),
    ])?;

    examples_menu.append_items(&[
        &QAndAPredefinedMenuItem::item(),
        &StarRepoPredefinedMenuItem::item(),
        &SummarizeNewsPredefinedMenuItem::item(),
    ])?;

    main_menu.append_items(&[&edit_menu, &window_menu, &application_menu, &examples_menu])?;

    let tailwind = include_str!("../public/tailwind.css");
    let cfg = dioxus::desktop::Config::default()
        .with_window(WindowBuilder::new().with_title("Floneum"))
        .with_menu(main_menu)
        .with_icon(Icon::from_rgba(include_bytes!("../public/Icon.rgba").to_vec(), 64, 64).unwrap())
        .with_custom_head(
            r#"
<style type="text/css">
    html, body {
        height: 100%;
        margin: 0;
        overscroll-behavior-y: none;
        overscroll-behavior-x: none;
        overflow: hidden;
        background-color: #ededf2;
    }
    #main, #bodywrap {
        height: 100%;
        margin: 0;
        overscroll-behavior-x: none;
        overscroll-behavior-y: none;
    }
</style>
<style type="text/css">
"#
            .to_owned()
                + tailwind
                + "</style>",
        );
    Ok(cfg)
}

pub fn use_apply_menu_event(state: Signal<ApplicationState>) {
    // let open_application = use_signal(|| None);
    // use_wry_event_handler(move |event, _| {
    //     if let dioxus::desktop::tao::event::Event::UserEvent(UserWindowEvent::MudaMenuEvent(muda_event)) = event {
    //         let menu_id = muda_event.menu_id;
    //         if menu_id == SavePredefinedMenuItem::id() {
    //             SavePredefinedMenuItem::save(&state.read());
    //         } else if menu_id == SaveAsPredefinedMenuItem::id() {
    //             SaveAsPredefinedMenuItem::save(&state.read());
    //         } else if menu_id == OpenPredefinedMenuItem::id() {
    //             OpenPredefinedMenuItem::open(open_application);
    //         } else if menu_id == QAndAPredefinedMenuItem::id() {
    //             QAndAPredefinedMenuItem::open(open_application);
    //         } else if menu_id == StarRepoPredefinedMenuItem::id() {
    //             StarRepoPredefinedMenuItem::open(open_application);
    //         } else if menu_id == SummarizeNewsPredefinedMenuItem::id() {
    //             SummarizeNewsPredefinedMenuItem::open(open_application);
    //         }
    //     }
    // });

    // if let Some(buffer) = open_application.take() {
    //     let as_str = std::str::from_utf8(&buffer).unwrap();
    //     if let Ok(from_storage) = serde_json::from_str(as_str) {
    //         state.set(from_storage);
    //     }
    // }
}

const SHORTCUT_LEADER: muda::accelerator::Modifiers = {
    #[cfg(target_os = "macos")]
    {
        muda::accelerator::Modifiers::SUPER
    }
    #[cfg(not(target_os = "macos"))]
    {
        muda::accelerator::Modifiers::CONTROL
    }
};

trait CustomMenuItem {
    fn name() -> &'static str;
    fn accelerator() -> Option<Accelerator>;
    fn id() -> MenuId {
        MenuId::new(Self::name())
    }
    fn item() -> MenuItem {
        MenuItem::new(Self::name(), true, Self::accelerator())
    }
}

struct SavePredefinedMenuItem;

impl CustomMenuItem for SavePredefinedMenuItem {
    fn name() -> &'static str {
        "Save"
    }

    fn accelerator() -> Option<Accelerator> {
        Accelerator::new(Some(SHORTCUT_LEADER), Code::KeyS).into()
    }
}

impl SavePredefinedMenuItem {
    fn save(state: &ApplicationState) {
        save_to_file(state, default_save_location());
    }
}

struct SaveAsPredefinedMenuItem;

impl CustomMenuItem for SaveAsPredefinedMenuItem {
    fn name() -> &'static str {
        "Save As"
    }

    fn accelerator() -> Option<Accelerator> {
        Accelerator::new(
            Some(SHORTCUT_LEADER | muda::accelerator::Modifiers::SHIFT),
            Code::KeyS,
        )
        .into()
    }
}

impl SaveAsPredefinedMenuItem {
    pub fn save(state: &ApplicationState) {
        if let Some(save_location) = rfd::FileDialog::new()
            .set_file_name("Floneum")
            .set_title("Save Location")
            .add_filter("Json", &["json"])
            .save_file()
        {
            save_to_file(state, save_location);
        }
    }
}

struct OpenPredefinedMenuItem;

impl CustomMenuItem for OpenPredefinedMenuItem {
    fn name() -> &'static str {
        "Open"
    }

    fn accelerator() -> Option<Accelerator> {
        Accelerator::new(Some(SHORTCUT_LEADER), Code::KeyO).into()
    }
}

impl OpenPredefinedMenuItem {
    pub fn open(mut state: Signal<Option<Vec<u8>>>) {
        if let Some(open_location) = rfd::FileDialog::new()
            .set_file_name("Floneum")
            .set_title("Open Location")
            .add_filter("Json", &["json"])
            .pick_file()
        {
            if let Ok(mut file) = File::open(open_location) {
                let mut buffer = Vec::new();

                if file.read_to_end(&mut buffer).is_ok() {
                    state.set(Some(buffer));
                }
            }
        }
    }
}

struct QAndAPredefinedMenuItem;

impl CustomMenuItem for QAndAPredefinedMenuItem {
    fn name() -> &'static str {
        "Open Q&A Example"
    }

    fn accelerator() -> Option<Accelerator> {
        None
    }
}

impl QAndAPredefinedMenuItem {
    pub fn open(mut state: Signal<Option<Vec<u8>>>) {
        let bytes = include_bytes!("../example_workflows/Q&A.json");
        state.set(Some(bytes.to_vec()));
    }
}

struct StarRepoPredefinedMenuItem;

impl CustomMenuItem for StarRepoPredefinedMenuItem {
    fn name() -> &'static str {
        "Open Star Repo Example"
    }

    fn accelerator() -> Option<Accelerator> {
        None
    }
}

impl StarRepoPredefinedMenuItem {
    pub fn open(mut state: Signal<Option<Vec<u8>>>) {
        let bytes = include_bytes!("../example_workflows/StarRepo.json");
        state.set(Some(bytes.to_vec()));
    }
}

struct SummarizeNewsPredefinedMenuItem;

impl CustomMenuItem for SummarizeNewsPredefinedMenuItem {
    fn name() -> &'static str {
        "Open Summarize News Example"
    }

    fn accelerator() -> Option<Accelerator> {
        None
    }
}

impl SummarizeNewsPredefinedMenuItem {
    pub fn open(mut state: Signal<Option<Vec<u8>>>) {
        let bytes = include_bytes!("../example_workflows/SummarizeNews.json");
        state.set(Some(bytes.to_vec()));
    }
}

fn default_save_location() -> PathBuf {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.push(SAVE_NAME);
    current_dir
}

fn save_to_file<D: Serialize>(data: &D, file: PathBuf) {
    match File::create(file) {
        Ok(mut file) => {
            log::info!("serializing");
            match serde_json::to_string(data) {
                Ok(bytes) => {
                    let _ = file.write_all(bytes.as_bytes());
                }
                Err(err) => {
                    log::error!("{}", err);
                }
            }
        }
        Err(err) => {
            log::error!("{}", err);
        }
    }
}
