use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use dioxus::prelude::ScopeState;
use dioxus_desktop::tao::accelerator::Accelerator;
use dioxus_desktop::tao::menu::{MenuBar, MenuId, MenuItem, MenuItemAttributes};
use dioxus_desktop::use_wry_event_handler;
use dioxus_desktop::{tao::window::Icon, WindowBuilder};
use dioxus_signals::{use_signal, Signal};
use serde::Serialize;

use crate::{ApplicationState, SAVE_NAME};

pub(crate) fn make_config() -> dioxus_desktop::Config {
    // Add a bunch of built-in menu items
    let mut main_menu = MenuBar::new();
    let mut edit_menu = MenuBar::new();
    let mut window_menu = MenuBar::new();
    let mut application_menu = MenuBar::new();

    edit_menu.add_native_item(MenuItem::Undo);
    edit_menu.add_native_item(MenuItem::Redo);
    edit_menu.add_native_item(MenuItem::Separator);
    edit_menu.add_native_item(MenuItem::Cut);
    edit_menu.add_native_item(MenuItem::Copy);
    edit_menu.add_native_item(MenuItem::Paste);
    edit_menu.add_native_item(MenuItem::SelectAll);

    window_menu.add_native_item(MenuItem::Quit);
    window_menu.add_native_item(MenuItem::Minimize);
    window_menu.add_native_item(MenuItem::Zoom);
    window_menu.add_native_item(MenuItem::Separator);
    window_menu.add_native_item(MenuItem::ShowAll);
    window_menu.add_native_item(MenuItem::EnterFullScreen);
    window_menu.add_native_item(MenuItem::Separator);
    window_menu.add_native_item(MenuItem::CloseWindow);

    application_menu.add_item(SaveMenuItem::item());
    application_menu.add_item(SaveAsMenuItem::item());
    application_menu.add_item(OpenMenuItem::item());

    main_menu.add_submenu("Floneum", true, application_menu);
    main_menu.add_submenu("Edit", true, edit_menu);
    main_menu.add_submenu("Window", true, window_menu);

    let tailwind = include_str!("../public/tailwind.css");
    dioxus_desktop::Config::default()
        .with_window(
            WindowBuilder::new()
                .with_title("Floneum")
                .with_menu(main_menu),
        )
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
        )
}

pub fn use_apply_menu_event(cx: &ScopeState, state: Signal<ApplicationState>) {
    let open_application = use_signal(cx, || None);
    use_wry_event_handler(cx, move |event, _| match event {
        dioxus_desktop::tao::event::Event::MenuEvent { menu_id, .. } => {
            let menu_id = *menu_id;
            if menu_id == SaveMenuItem::id() {
                SaveMenuItem::save(&state.read());
            } else if menu_id == SaveAsMenuItem::id() {
                SaveAsMenuItem::save(&state.read());
            } else if menu_id == OpenMenuItem::id() {
                OpenMenuItem::open(open_application);
            }
        }
        _ => {}
    });

    if let Some(buffer) = open_application.take() {
        let as_str = std::str::from_utf8(&buffer).unwrap();
        if let Ok(from_storage) = serde_json::from_str(as_str) {
            state.set(from_storage);
        }
    }
}

const SHORTCUT_LEADER: dioxus_desktop::tao::keyboard::ModifiersState = {
    #[cfg(target_os = "macos")]
    {
        dioxus_desktop::tao::keyboard::ModifiersState::SUPER
    }
    #[cfg(not(target_os = "macos"))]
    {
        dioxus_desktop::tao::keyboard::ModifiersState::CONTROL
    }
};

struct SaveMenuItem;

impl SaveMenuItem {
    fn name() -> &'static str {
        "Save"
    }

    pub fn id() -> MenuId {
        MenuId::new(Self::name())
    }

    pub fn item() -> MenuItemAttributes<'static> {
        MenuItemAttributes::new(Self::name())
            .with_id(Self::id())
            .with_accelerators(&Accelerator::new(
                SHORTCUT_LEADER,
                dioxus_desktop::tao::keyboard::KeyCode::KeyS,
            ))
    }

    pub fn save(state: &ApplicationState) {
        save_to_file(state, default_save_location());
    }
}

struct SaveAsMenuItem;

impl SaveAsMenuItem {
    fn name() -> &'static str {
        "Save As"
    }

    pub fn id() -> MenuId {
        MenuId::new(Self::name())
    }

    pub fn item() -> MenuItemAttributes<'static> {
        MenuItemAttributes::new(Self::name())
            .with_id(Self::id())
            .with_accelerators(&Accelerator::new(
                SHORTCUT_LEADER | dioxus_desktop::tao::keyboard::ModifiersState::SHIFT,
                dioxus_desktop::tao::keyboard::KeyCode::KeyS,
            ))
    }

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

struct OpenMenuItem;

impl OpenMenuItem {
    fn name() -> &'static str {
        "Open"
    }

    pub fn id() -> MenuId {
        MenuId::new(Self::name())
    }

    pub fn item() -> MenuItemAttributes<'static> {
        MenuItemAttributes::new(Self::name())
            .with_id(Self::id())
            .with_accelerators(&Accelerator::new(
                SHORTCUT_LEADER,
                dioxus_desktop::tao::keyboard::KeyCode::KeyO,
            ))
    }

    pub fn open(state: Signal<Option<Vec<u8>>>) {
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
