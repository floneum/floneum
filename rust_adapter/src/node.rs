use crate::{plugins::main::types::NodeId, IntoPrimitiveValue, PrimitiveValue};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Node {
    id: NodeId,
}

impl From<NodeId> for Node {
    fn from(id: NodeId) -> Self {
        Node { id }
    }
}

impl Node {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn screenshot(&self) -> DynamicImage {
        let screenshot = crate::plugins::main::imports::screenshot_element(self.id);
        image::load(std::io::Cursor::new(&*screenshot), image::ImageFormat::Jpeg).unwrap()
    }

    pub fn click(&self) {
        crate::plugins::main::imports::click_element(self.id);
    }

    pub fn wait_for_element(&self, selector: &str) -> Node {
        crate::plugins::main::imports::find_child_of_element(self.id, selector).into()
    }

    pub fn outer_html(&self) -> String {
        crate::plugins::main::imports::get_element_outer_html(self.id)
    }

    pub fn text(&self) -> String {
        crate::plugins::main::imports::get_element_text(self.id)
    }
}

impl IntoPrimitiveValue for Node {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Node(self.id)
    }
}
