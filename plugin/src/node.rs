use crate::host::State;
use crate::plugins::main;

use crate::plugins::main::types::Node;

use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostNode for State {
    async fn get_element_text(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn click_element(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn type_into_element(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
        _keys: String,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn get_element_outer_html(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<String> {
        todo!()
    }

    async fn screenshot_element(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    async fn find_child_of_element(
        &mut self,
        _self_: wasmtime::component::Resource<Node>,
        _query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    fn drop(&mut self, _rep: wasmtime::component::Resource<Node>) -> wasmtime::Result<()> {
        todo!()
    }
}
