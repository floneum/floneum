use crate::host::State;
use crate::plugins::main;

use crate::plugins::main::types::{Node, Page};

use wasmtime::component::__internal::async_trait;

#[async_trait]
impl main::types::HostPage for State {
    async fn new(
        &mut self,
        _mode: main::types::BrowserMode,
        _url: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Page>> {
        todo!()
    }

    async fn find_in_current_page(
        &mut self,
        _self_: wasmtime::component::Resource<Page>,
        _query: String,
    ) -> wasmtime::Result<wasmtime::component::Resource<Node>> {
        todo!()
    }

    async fn screenshot_browser(
        &mut self,
        _self_: wasmtime::component::Resource<Page>,
    ) -> wasmtime::Result<Vec<u8>> {
        todo!()
    }

    fn drop(&mut self, _rep: wasmtime::component::Resource<Page>) -> wasmtime::Result<()> {
        todo!()
    }
}
