mod host;
mod plugin;
mod proxies;

wasmtime::component::bindgen!({
    path: "../wit",
    async: true,
});
