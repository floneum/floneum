pub use crate::exports::plugins::main::definitions::Guest;
pub use crate::plugins::main::imports::log_to_user;
pub use crate::plugins::main::types::*;

pub struct Page {
    page: PageResource,
}

impl From<PageResource> for Page {
    fn from(page: PageResource) -> Self {
        Self { page }
    }
}

impl Page {
    pub fn new(mode: BrowserMode, url: &str) -> Self {
        Self {
            page: create_page(mode, url),
        }
    }

    pub fn html(&self) -> String {
        page_html(self.page)
    }

    pub fn screenshot(&self) -> Vec<u8> {
        screenshot_browser(self.page)
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        drop_page(self.page);
    }
}

pub struct Node {
    node: NodeResource,
}

impl From<NodeResource> for Node {
    fn from(node: NodeResource) -> Self {
        Self { node }
    }
}

impl Node {
    pub fn new(node: NodeResource) -> Self {
        Self { node }
    }

    pub fn get_text(&self) -> String {
        get_element_text(self.node)
    }

    pub fn click(&self) {
        click_element(self.node);
    }

    pub fn send_keys(&self, keys: &str) {
        type_into_element(self.node, keys);
    }

    pub fn outer_html(&self) -> String {
        get_element_outer_html(self.node)
    }

    pub fn screenshot_element(&self) -> Vec<u8> {
        screenshot_element(self.node)
    }

    pub fn find_child(&self, selector: &str) -> Node {
        let resource = find_child_of_element(self.node, selector);
        Node { node: resource }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        drop_node(self.node);
    }
}

pub struct EmbeddingDb {
    db: EmbeddingDbResource,
}

impl From<EmbeddingDbResource> for EmbeddingDb {
    fn from(db: EmbeddingDbResource) -> Self {
        Self { db }
    }
}

impl EmbeddingDb {
    pub fn new(embeddings: &[Embedding], documents: &[String]) -> Self {
        Self {
            db: create_embedding_db(embeddings, documents),
        }
    }

    pub fn add_embedding(&self, embedding: &Embedding, document: &str) {
        add_embedding(self.db, embedding, document);
    }

    pub fn find_closest_documents(&self, search: &Embedding, count: u32) -> Vec<String> {
        find_closest_documents(self.db, search, count)
    }
}

impl Drop for EmbeddingDb {
    fn drop(&mut self) {
        drop_embedding_db(self.db);
    }
}

pub struct TextGenerationModel {
    model: TextGenerationModelResource,
}

impl From<TextGenerationModelResource> for TextGenerationModel {
    fn from(model: TextGenerationModelResource) -> Self {
        Self { model }
    }
}

impl TextGenerationModel {
    pub fn new(model: ModelType) -> Self {
        let model = create_model(model);
        Self { model }
    }

    pub fn model_downloaded(model: ModelType) -> bool {
        text_generation_model_downloaded(model)
    }

    pub fn infer(&self, input: &str, max_tokens: Option<u32>, stop_on: Option<&str>) -> String {
        infer(self.model, input, max_tokens, stop_on)
    }

    pub fn infer_structured(&self, input: &str, regex: &str) -> String {
        infer_structured(self.model, input, regex)
    }
}

impl Drop for TextGenerationModel {
    fn drop(&mut self) {
        drop_model(self.model);
    }
}

pub struct EmbeddingModel {
    model: EmbeddingModelResource,
}

impl From<EmbeddingModelResource> for EmbeddingModel {
    fn from(model: EmbeddingModelResource) -> Self {
        Self { model }
    }
}

impl EmbeddingModel {
    pub fn new(model: EmbeddingModelType) -> Self {
        let model = create_embedding_model(model);
        Self { model }
    }

    pub fn model_downloaded(model: EmbeddingModelType) -> bool {
        embedding_model_downloaded(model)
    }

    pub fn get_embedding(&self, document: &str) -> Embedding {
        get_embedding(self.model, document)
    }
}

impl Drop for EmbeddingModel {
    fn drop(&mut self) {
        drop_embedding_model(self.model);
    }
}

pub trait IntoInputValue<T = ()> {
    fn into_input_value(self) -> Vec<PrimitiveValue>;
}

pub trait IntoReturnValue<T = ()> {
    fn into_return_value(self) -> Vec<PrimitiveValue>;
}

impl<T: IntoPrimitiveValue> IntoReturnValue for T {
    fn into_return_value(self) -> Vec<PrimitiveValue> {
        vec![self.into_primitive_value()]
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for T {
    fn into_input_value(self) -> Vec<PrimitiveValue> {
        vec![self.into_primitive_value()]
    }
}

#[doc(hidden)]
pub struct VecMarker;

impl<T: IntoPrimitiveValue> IntoReturnValue<VecMarker> for Vec<T> {
    fn into_return_value(self) -> Vec<PrimitiveValue> {
        self.into_iter().map(|x| x.into_primitive_value()).collect()
    }
}

impl<T: IntoPrimitiveValue> IntoInputValue for Vec<T> {
    fn into_input_value(self) -> Vec<PrimitiveValue> {
        self.into_iter().map(|x| x.into_primitive_value()).collect()
    }
}

trait IntoPrimitiveValue {
    fn into_primitive_value(self) -> PrimitiveValue;
}

impl IntoPrimitiveValue for PrimitiveValue {
    fn into_primitive_value(self) -> PrimitiveValue {
        self
    }
}

impl IntoPrimitiveValue for ModelType {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::ModelType(self)
    }
}

impl IntoPrimitiveValue for TextGenerationModel {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Model(self.model)
    }
}

impl IntoPrimitiveValue for EmbeddingModel {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::EmbeddingModel(self.model)
    }
}

impl IntoPrimitiveValue for EmbeddingModelType {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::EmbeddingModelType(self)
    }
}

impl IntoPrimitiveValue for Node {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Node(self.node)
    }
}

impl IntoPrimitiveValue for Page {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Page(self.page)
    }
}

impl IntoPrimitiveValue for bool {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Boolean(self)
    }
}

impl IntoPrimitiveValue for i64 {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Number(self)
    }
}

impl IntoPrimitiveValue for f64 {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Float(self)
    }
}

impl IntoPrimitiveValue for String {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Text(self)
    }
}

impl IntoPrimitiveValue for Embedding {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Embedding(self)
    }
}

impl IntoPrimitiveValue for EmbeddingDb {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Database(self.db)
    }
}

pub trait IntoReturnValues<T = ()> {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>>;
}

impl<T: IntoReturnValue<I>, I> IntoReturnValues<I> for T {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        vec![self.into_return_value()]
    }
}

#[doc(hidden)]
pub struct OptionMarker;

impl<T: IntoReturnValue> IntoReturnValue<OptionMarker> for Option<T> {
    fn into_return_value(self) -> Vec<PrimitiveValue> {
        match self {
            Some(value) => value.into_return_value(),
            None => vec![],
        }
    }
}

#[doc(hidden)]
pub struct UnitMarker;

impl IntoReturnValues<UnitMarker> for () {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        vec![]
    }
}

impl<A: IntoReturnValue<A2>, A2> IntoReturnValues<(A2,)> for (A,) {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a,) = self;
        vec![a.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2> IntoReturnValues<(A2, B2)> for (A, B) {
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b) = self;
        vec![a.into_return_value(), b.into_return_value()]
    }
}

impl<A: IntoReturnValue<A2>, A2, B: IntoReturnValue<B2>, B2, C: IntoReturnValue<C2>, C2>
    IntoReturnValues<(A2, B2, C2)> for (A, B, C)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
        ]
    }
}

impl<
        A: IntoReturnValue<A2>,
        A2,
        B: IntoReturnValue<B2>,
        B2,
        C: IntoReturnValue<C2>,
        C2,
        D: IntoReturnValue<D2>,
        D2,
    > IntoReturnValues<(A2, B2, C2, D2)> for (A, B, C, D)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c, d) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
            d.into_return_value(),
        ]
    }
}

impl<
        A: IntoReturnValue<A2>,
        A2,
        B: IntoReturnValue<B2>,
        B2,
        C: IntoReturnValue<C2>,
        C2,
        D: IntoReturnValue<D2>,
        D2,
        E: IntoReturnValue<E2>,
        E2,
    > IntoReturnValues<(A2, B2, C2, D2, E2)> for (A, B, C, D, E)
{
    fn into_return_values(self) -> Vec<Vec<PrimitiveValue>> {
        let (a, b, c, d, e) = self;
        vec![
            a.into_return_value(),
            b.into_return_value(),
            c.into_return_value(),
            d.into_return_value(),
            e.into_return_value(),
        ]
    }
}

/// A wrapper around a file path.
pub struct File(std::path::PathBuf);

impl std::ops::Deref for File {
    type Target = std::path::PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<std::path::PathBuf> for File {
    fn from(path: std::path::PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for File {
    fn from(path: String) -> Self {
        Self(std::path::PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for File {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::File(self.0.display().to_string())
    }
}

/// A wrapper around a folder path.
pub struct Folder(std::path::PathBuf);

impl std::ops::Deref for Folder {
    type Target = std::path::PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<std::path::PathBuf> for Folder {
    fn from(path: std::path::PathBuf) -> Self {
        Self(path)
    }
}

impl From<String> for Folder {
    fn from(path: String) -> Self {
        Self(std::path::PathBuf::from(path))
    }
}

impl IntoPrimitiveValue for Folder {
    fn into_primitive_value(self) -> PrimitiveValue {
        PrimitiveValue::Folder(self.0.display().to_string())
    }
}
