use crate::{
    plugins::main::{
        imports::{new_tab, remove_tab},
        types::TabId,
    },
    IntoPrimitiveValue, Node, PrimitiveValue,
};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Tab {
    id: TabId,
    drop: bool,
}

impl From<TabId> for Tab {
    fn from(id: TabId) -> Self {
        Tab { id, drop: false }
    }
}

impl Drop for Tab {
    fn drop(&mut self) {
        if self.drop {
            println!("removing {:?}", self.id);
            remove_tab(self.id);
        }
    }
}

impl Tab {
    pub fn new(headless: bool) -> Self {
        let id = new_tab(headless);
        Tab { id, drop: true }
    }

    pub fn leak(self) -> TabId {
        let id = self.id;
        std::mem::forget(self);
        id
    }

    pub fn id(&self) -> TabId {
        self.id
    }

    pub fn screenshot(&self) -> DynamicImage {
        let screenshot = crate::plugins::main::imports::screenshot_browser(self.id);
        image::load(std::io::Cursor::new(&*screenshot), image::ImageFormat::Jpeg).unwrap()
    }

    pub fn goto(&self, url: &str) {
        crate::plugins::main::imports::browse_to(self.id, url);
    }

    pub fn wait_for_element(&self, selector: &str) -> Node {
        crate::plugins::main::imports::find_in_current_page(self.id, selector).into()
    }
}

impl IntoPrimitiveValue for Tab {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Tab(self.id)
    }
}

impl IntoPrimitiveValue for TabId {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Tab(self)
    }
}
