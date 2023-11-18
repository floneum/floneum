use dioxus::prelude::*;
use dioxus_signals::Signal;
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Theme {
    border: Color,
    background: Color,
    text: Color,
    foreground: Color,
    foreground_hover: Color,
}

#[allow(unused)]
impl Theme {
    pub fn current() -> Signal<Theme> {
        match consume_context::<Signal<Theme>>() {
            Some(theme) => theme,
            None => provide_root_context(Signal::new_in_scope(Theme::DARK, ScopeId::ROOT)).unwrap(),
        }
    }

    pub const DARK: Theme = Theme {
        border: Color("border-[#6A7C93]"),
        text: Color("text-blue-50"),
        background: Color("fill-zinc-700"),
        foreground: Color("bg-[#313943]"),
        foreground_hover: Color("hover:bg-[#475362]"),
    };

    pub const WHITE: Theme = Theme {
        border: Color("border-black"),
        text: Color("text-black"),
        background: Color("fill-[#d4dff2]"),
        foreground: Color("bg-[#b8c5da]"),
        foreground_hover: Color("hover:bg-[#dae1ec]"),
    };
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Color(&'static str);

impl Color {
    pub fn outline_color() -> Color {
        Theme::current().read().border
    }

    pub fn text_color() -> Color {
        Theme::current().read().text
    }

    pub fn background_color() -> Color {
        Theme::current().read().background
    }

    pub fn foreground_color() -> Color {
        Theme::current().read().foreground
    }

    pub fn foreground_hover() -> Color {
        Theme::current().read().foreground_hover
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
