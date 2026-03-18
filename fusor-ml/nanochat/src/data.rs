use std::{collections::HashSet, fs, path::Path};

use fusor::VarBuilder;
use fusor_gguf::GgufValue;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::config::RuntimeConfig;

const BOS_TOKEN: &str = "<bos>";
const EOT_TOKEN: &str = "<eot>";
const STEP_LEVELS: u8 = 2;
const CANVAS_SIZE: f32 = 128.0;
const CANVAS_PADDING: f32 = 14.0;

#[derive(Clone)]
pub struct StrokeTokenizer {
    tokens: Vec<String>,
    bos_token: u32,
    eot_token: u32,
    move_start: u32,
}

pub struct SourceDataset {
    files: Vec<SourceFile>,
}

#[derive(Clone)]
pub struct SourceFile {
    path: String,
    tokens: Vec<u32>,
    prompt_tokens: Vec<u32>,
    target_tokens: Vec<u32>,
}

pub struct DatasetSplit {
    pub train: SourceDataset,
    pub validation: SourceDataset,
    pub test: SourceDataset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Bos,
    Eot,
    Move(Direction, u8),
}

#[derive(Clone)]
struct RenderState {
    cursor: (i32, i32),
    current_stroke: Vec<(i32, i32)>,
    completed_strokes: Vec<Vec<(i32, i32)>>,
}

#[derive(Clone, Copy)]
enum ShapeKind {
    Circle,
    Square,
    Rectangle,
    Triangle,
    Diamond,
    Hexagon,
    Octagon,
    House,
    Trapezoid,
    Parallelogram,
}

pub struct Batch {
    pub windows: Vec<Vec<u32>>,
    pub masks: Vec<Vec<f32>>,
    pub valid_tokens: f32,
}

impl Direction {
    fn all() -> [Self; 8] {
        [
            Self::N,
            Self::NE,
            Self::E,
            Self::SE,
            Self::S,
            Self::SW,
            Self::W,
            Self::NW,
        ]
    }

    fn index(self) -> usize {
        match self {
            Self::N => 0,
            Self::NE => 1,
            Self::E => 2,
            Self::SE => 3,
            Self::S => 4,
            Self::SW => 5,
            Self::W => 6,
            Self::NW => 7,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::N => "N",
            Self::NE => "NE",
            Self::E => "E",
            Self::SE => "SE",
            Self::S => "S",
            Self::SW => "SW",
            Self::W => "W",
            Self::NW => "NW",
        }
    }

    fn delta(self, step: u8) -> (i32, i32) {
        let step = step as i32;
        match self {
            Self::N => (0, -step),
            Self::NE => (step, -step),
            Self::E => (step, 0),
            Self::SE => (step, step),
            Self::S => (0, step),
            Self::SW => (-step, step),
            Self::W => (-step, 0),
            Self::NW => (-step, -step),
        }
    }

    fn rotate(self, steps: usize) -> Self {
        Self::all()[(self.index() + steps) % Self::all().len()]
    }

    fn mirror_horizontal(self) -> Self {
        match self {
            Self::N => Self::N,
            Self::NE => Self::NW,
            Self::E => Self::W,
            Self::SE => Self::SW,
            Self::S => Self::S,
            Self::SW => Self::SE,
            Self::W => Self::E,
            Self::NW => Self::NE,
        }
    }

    fn opposite(self) -> Self {
        self.rotate(Direction::all().len() / 2)
    }
}

impl ShapeKind {
    fn all() -> [Self; 10] {
        [
            Self::Circle,
            Self::Square,
            Self::Rectangle,
            Self::Triangle,
            Self::Diamond,
            Self::Hexagon,
            Self::Octagon,
            Self::House,
            Self::Trapezoid,
            Self::Parallelogram,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Circle => "circle",
            Self::Square => "square",
            Self::Rectangle => "rectangle",
            Self::Triangle => "triangle",
            Self::Diamond => "diamond",
            Self::Hexagon => "hexagon",
            Self::Octagon => "octagon",
            Self::House => "house",
            Self::Trapezoid => "trapezoid",
            Self::Parallelogram => "parallelogram",
        }
    }

    fn is_closed(self) -> bool {
        true
    }
}

impl StrokeTokenizer {
    pub fn new() -> Self {
        let mut tokens = vec![BOS_TOKEN.to_string(), EOT_TOKEN.to_string()];
        let bos_token = 0;
        let eot_token = 1;
        let move_start = tokens.len() as u32;

        for direction in Direction::all() {
            for step in 1..=STEP_LEVELS {
                tokens.push(format!("MOVE_{}_{}", direction.as_str(), step));
            }
        }

        Self {
            tokens,
            bos_token,
            eot_token,
            move_start,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> u32 {
        self.bos_token
    }

    pub fn eot_token(&self) -> u32 {
        self.eot_token
    }

    pub fn token_name(&self, token: u32) -> &str {
        self.tokens
            .get(token as usize)
            .map(String::as_str)
            .unwrap_or("<invalid>")
    }

    pub fn describe_tokens(&self, tokens: &[u32], limit: usize) -> String {
        let shown = tokens
            .iter()
            .take(limit)
            .map(|token| self.token_name(*token))
            .collect::<Vec<_>>()
            .join(" ");
        if tokens.len() > limit {
            format!("{shown} ...")
        } else {
            shown
        }
    }

    pub fn gguf_metadata(&self) -> Vec<(String, GgufValue)> {
        let token_types: Box<[GgufValue]> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index as u32 == self.bos_token || index as u32 == self.eot_token {
                    GgufValue::U32(3)
                } else {
                    GgufValue::U32(1)
                }
            })
            .collect();

        vec![
            (
                "tokenizer.ggml.model".to_string(),
                GgufValue::String("word".into()),
            ),
            (
                "tokenizer.ggml.pre".to_string(),
                GgufValue::String("default".into()),
            ),
            (
                "tokenizer.ggml.add_bos_token".to_string(),
                GgufValue::Bool(false),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    self.tokens
                        .iter()
                        .cloned()
                        .map(|token| GgufValue::String(token.into_boxed_str()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                GgufValue::Array(token_types),
            ),
            (
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::U32(self.bos_token),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::U32(self.eot_token),
            ),
        ]
    }

    pub fn from_var_builder(vb: &VarBuilder) -> fusor::Result<Self> {
        let tokenizer = Self::new();
        let tokens = vb
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.tokens metadata missing"))?
            .to_array()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let loaded_tokens = tokens
            .iter()
            .map(|token| {
                token
                    .to_string()
                    .map(String::from)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(fusor::Error::msg)?;
        if loaded_tokens != tokenizer.tokens {
            return Err(fusor::Error::msg(
                "GGUF tokenizer vocabulary does not match the expected stroke vocabulary",
            ));
        }

        let bos_token = vb
            .get_metadata("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.bos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        let eot_token = vb
            .get_metadata("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| fusor::Error::msg("tokenizer.ggml.eos_token_id metadata missing"))?
            .to_u32()
            .map_err(|error| fusor::Error::msg(error.to_string()))?;
        if bos_token != tokenizer.bos_token || eot_token != tokenizer.eot_token {
            return Err(fusor::Error::msg(
                "GGUF tokenizer special token ids do not match the expected stroke tokenizer ids",
            ));
        }

        Ok(tokenizer)
    }

    fn move_token(&self, direction: Direction, step: u8) -> u32 {
        assert!((1..=STEP_LEVELS).contains(&step));
        self.move_start + (direction.index() * STEP_LEVELS as usize + (step as usize - 1)) as u32
    }

    fn decode_token_kind(&self, token: u32) -> TokenKind {
        if token == self.bos_token {
            return TokenKind::Bos;
        }
        if token == self.eot_token {
            return TokenKind::Eot;
        }

        let move_token_count = Direction::all().len() as u32 * STEP_LEVELS as u32;
        let move_end = self.move_start + move_token_count;
        if (self.move_start..move_end).contains(&token) {
            let offset = token - self.move_start;
            let direction = Direction::all()[(offset / STEP_LEVELS as u32) as usize];
            let step = (offset % STEP_LEVELS as u32) as u8 + 1;
            return TokenKind::Move(direction, step);
        }

        panic!("unknown stroke token id {token}");
    }
}

impl SourceDataset {
    pub fn num_docs(&self) -> usize {
        self.files.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.files.iter().map(|file| file.tokens.len()).sum()
    }

    pub fn num_training_windows(&self, _block_size: usize) -> usize {
        self.files.len()
    }

    pub fn max_tokens_per_example(&self) -> usize {
        self.files
            .iter()
            .map(|file| file.tokens.len())
            .max()
            .unwrap_or(0)
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn sample_batch(&self, rng: &mut StdRng, pad_token: u32, config: &RuntimeConfig) -> Batch {
        if self.files.is_empty() {
            return Batch {
                windows: Vec::new(),
                masks: Vec::new(),
                valid_tokens: 0.0,
            };
        }

        let sampled = (0..config.batch_size)
            .map(|_| {
                let file_index = rng.random_range(0..self.files.len());
                self.sample_window_at(file_index, pad_token, config.block_size)
            })
            .collect::<Vec<_>>();

        Batch {
            windows: sampled
                .iter()
                .map(|(window, _, _)| window.clone())
                .collect(),
            masks: sampled.iter().map(|(_, mask, _)| mask.clone()).collect(),
            valid_tokens: sampled.iter().map(|(_, _, valid)| *valid).sum(),
        }
    }

    pub fn epoch_batches(
        &self,
        rng: &mut StdRng,
        pad_token: u32,
        config: &RuntimeConfig,
    ) -> Vec<Batch> {
        if self.files.is_empty() {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..self.files.len()).collect();
        indices.shuffle(rng);

        let mut batches = Vec::with_capacity(indices.len().div_ceil(config.batch_size));
        for chunk in indices.chunks(config.batch_size) {
            let sampled: Vec<_> = chunk
                .iter()
                .map(|&index| self.sample_window_at(index, pad_token, config.block_size))
                .collect();
            batches.push(Batch {
                windows: sampled
                    .iter()
                    .map(|(window, _, _)| window.clone())
                    .collect(),
                masks: sampled.iter().map(|(_, mask, _)| mask.clone()).collect(),
                valid_tokens: sampled.iter().map(|(_, _, valid)| *valid).sum(),
            });
        }
        batches
    }

    pub fn steps_per_epoch(&self, _block_size: usize, batch_size: usize) -> usize {
        self.files.len().div_ceil(batch_size).max(1)
    }

    pub fn evaluation_batches(&self, pad_token: u32, config: &RuntimeConfig) -> Vec<Batch> {
        if self.files.is_empty() {
            return Vec::new();
        }

        let steps = config
            .eval_batches
            .saturating_mul(config.batch_size)
            .min(self.files.len());
        let mut batches = Vec::new();
        let mut current_windows = Vec::new();
        let mut current_masks = Vec::new();
        let mut current_valid_tokens = 0.0;

        for file_index in 0..steps {
            let (window, mask, valid_tokens) =
                self.sample_window_at(file_index, pad_token, config.block_size);
            current_windows.push(window);
            current_masks.push(mask);
            current_valid_tokens += valid_tokens;

            if current_windows.len() == config.batch_size {
                batches.push(Batch {
                    windows: std::mem::take(&mut current_windows),
                    masks: std::mem::take(&mut current_masks),
                    valid_tokens: current_valid_tokens,
                });
                current_valid_tokens = 0.0;
            }
        }

        if !current_windows.is_empty() {
            batches.push(Batch {
                windows: current_windows,
                masks: current_masks,
                valid_tokens: current_valid_tokens,
            });
        }

        batches
    }

    fn sample_window_at(
        &self,
        file_index: usize,
        pad_token: u32,
        block_size: usize,
    ) -> (Vec<u32>, Vec<f32>, f32) {
        let file = &self.files[file_index];
        let start = file.tokens.len().saturating_sub(block_size + 1);
        let slice = &file.tokens[start..];

        let mut window = vec![pad_token; block_size + 1];
        let mut mask = vec![0.0; block_size];
        window[..slice.len()].copy_from_slice(slice);

        let mut valid = 0.0;
        for local_index in 0..slice.len().saturating_sub(1) {
            mask[local_index] = 1.0;
            valid += 1.0;
        }

        (window, mask, valid)
    }
}

impl SourceFile {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn target_tokens(&self) -> &[u32] {
        &self.target_tokens
    }

    pub fn completion_prompt_tokens(
        &self,
        prompt_tokens: usize,
        block_size: usize,
        sample_tokens: usize,
    ) -> &[u32] {
        let max_prompt_tokens = prompt_tokens.min(block_size.saturating_sub(sample_tokens + 1));
        &self.prompt_tokens[..self.prompt_tokens.len().min(max_prompt_tokens)]
    }
}

pub fn synthetic_dataset_split(
    tokenizer: &StrokeTokenizer,
    config: &RuntimeConfig,
) -> DatasetSplit {
    DatasetSplit {
        train: generate_split(
            tokenizer,
            config.train_examples,
            config.seed ^ 0xA11CE,
            "train",
        ),
        validation: generate_split(
            tokenizer,
            config.validation_examples,
            config.seed ^ 0xBEEFu64,
            "valid",
        ),
        test: generate_split(
            tokenizer,
            config.test_examples,
            config.seed ^ 0xC0DEu64,
            "test",
        ),
    }
}

pub fn write_tokens_to_svg_file(
    tokenizer: &StrokeTokenizer,
    prompt_tokens: &[u32],
    continuation_tokens: &[u32],
    path: &Path,
) {
    let mut prompt_state = RenderState::default();
    render_tokens_into_state(tokenizer, prompt_tokens, &mut prompt_state);
    prompt_state.finish_current_stroke();
    let prompt_strokes = prompt_state.completed_strokes.clone();

    let mut continuation_state = RenderState {
        cursor: prompt_state.cursor,
        current_stroke: prompt_state.current_stroke,
        completed_strokes: Vec::new(),
    };
    render_tokens_into_state(tokenizer, continuation_tokens, &mut continuation_state);
    continuation_state.finish_current_stroke();

    let svg = svg_document(&prompt_strokes, &continuation_state.completed_strokes);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create SVG output directory {}: {error}",
                parent.display()
            )
        });
    }
    fs::write(path, svg)
        .unwrap_or_else(|error| panic!("failed to write SVG sample {}: {error}", path.display()));
}

pub fn windows_to_token_inputs(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows
        .iter()
        .map(|window| window[..window.len().saturating_sub(1)].to_vec())
        .collect()
}

pub fn windows_to_token_targets(windows: &[Vec<u32>]) -> Vec<Vec<u32>> {
    windows.iter().map(|window| window[1..].to_vec()).collect()
}

pub fn autoregressive_context(
    tokens: &[u32],
    pad_token: u32,
    block_size: usize,
) -> (Vec<u32>, usize) {
    let mut context = vec![pad_token; block_size];
    let slice = if tokens.len() > block_size {
        &tokens[tokens.len() - block_size..]
    } else {
        tokens
    };
    context[..slice.len()].copy_from_slice(slice);
    (context, slice.len().saturating_sub(1))
}

pub fn position_indexes(batch_size: usize, block_size: usize) -> Vec<Vec<u32>> {
    (0..batch_size)
        .map(|_| (0..block_size).map(|position| position as u32).collect())
        .collect()
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            cursor: (0, 0),
            current_stroke: Vec::new(),
            completed_strokes: Vec::new(),
        }
    }
}

impl RenderState {
    fn finish_current_stroke(&mut self) {
        if self.current_stroke.len() > 1 {
            self.completed_strokes
                .push(std::mem::take(&mut self.current_stroke));
        } else {
            self.current_stroke.clear();
        }
    }
}

fn render_tokens_into_state(tokenizer: &StrokeTokenizer, tokens: &[u32], state: &mut RenderState) {
    for token in tokens {
        match tokenizer.decode_token_kind(*token) {
            TokenKind::Bos | TokenKind::Eot => {}
            TokenKind::Move(direction, step) => {
                if state.current_stroke.is_empty() {
                    state.current_stroke.push(state.cursor);
                }
                let (dx, dy) = direction.delta(step);
                state.cursor.0 += dx;
                state.cursor.1 += dy;
                state.current_stroke.push(state.cursor);
            }
        }
    }
}

fn svg_document(
    prompt_strokes: &[Vec<(i32, i32)>],
    continuation_strokes: &[Vec<(i32, i32)>],
) -> String {
    let all_points = prompt_strokes
        .iter()
        .chain(continuation_strokes.iter())
        .flat_map(|stroke| stroke.iter().copied())
        .collect::<Vec<_>>();

    let (offset_x, offset_y, scale) = if all_points.is_empty() {
        (CANVAS_SIZE / 2.0, CANVAS_SIZE / 2.0, 1.0)
    } else {
        let min_x = all_points.iter().map(|(x, _)| *x).min().unwrap() as f32;
        let max_x = all_points.iter().map(|(x, _)| *x).max().unwrap() as f32;
        let min_y = all_points.iter().map(|(_, y)| *y).min().unwrap() as f32;
        let max_y = all_points.iter().map(|(_, y)| *y).max().unwrap() as f32;
        let width = (max_x - min_x).max(1.0);
        let height = (max_y - min_y).max(1.0);
        let scale = ((CANVAS_SIZE - CANVAS_PADDING * 2.0) / width)
            .min((CANVAS_SIZE - CANVAS_PADDING * 2.0) / height);
        let scaled_width = width * scale;
        let scaled_height = height * scale;
        let offset_x = CANVAS_PADDING + (CANVAS_SIZE - CANVAS_PADDING * 2.0 - scaled_width) / 2.0;
        let offset_y = CANVAS_PADDING + (CANVAS_SIZE - CANVAS_PADDING * 2.0 - scaled_height) / 2.0;
        (offset_x - min_x * scale, offset_y - min_y * scale, scale)
    };

    let prompt_paths = prompt_strokes
        .iter()
        .filter_map(|stroke| polyline_path(stroke, offset_x, offset_y, scale))
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#264653\" stroke-width=\"5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.65\"/>"
            )
        })
        .collect::<Vec<_>>()
        .join("");
    let continuation_paths = continuation_strokes
        .iter()
        .filter_map(|stroke| polyline_path(stroke, offset_x, offset_y, scale))
        .map(|path| {
            format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#e76f51\" stroke-width=\"5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
            )
        })
        .collect::<Vec<_>>()
        .join("");

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {CANVAS_SIZE} {CANVAS_SIZE}\"><rect width=\"{CANVAS_SIZE}\" height=\"{CANVAS_SIZE}\" rx=\"18\" fill=\"#fdf6e3\"/><rect x=\"6\" y=\"6\" width=\"{inner}\" height=\"{inner}\" rx=\"14\" fill=\"#fffaf0\" stroke=\"#e9dcc3\" stroke-width=\"2\"/>{prompt_paths}{continuation_paths}</svg>",
        inner = CANVAS_SIZE - 12.0,
    )
}

fn polyline_path(
    stroke: &[(i32, i32)],
    offset_x: f32,
    offset_y: f32,
    scale: f32,
) -> Option<String> {
    let mut points = stroke.iter();
    let &(first_x, first_y) = points.next()?;
    let mut path = format!(
        "M {:.2} {:.2}",
        offset_x + first_x as f32 * scale,
        offset_y + first_y as f32 * scale
    );
    for &(x, y) in points {
        path.push_str(&format!(
            " L {:.2} {:.2}",
            offset_x + x as f32 * scale,
            offset_y + y as f32 * scale
        ));
    }
    Some(path)
}

fn generate_split(
    tokenizer: &StrokeTokenizer,
    example_count: usize,
    seed: u64,
    split_name: &str,
) -> SourceDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shapes = (0..example_count)
        .map(|index| ShapeKind::all()[index % ShapeKind::all().len()])
        .collect::<Vec<_>>();
    shapes.shuffle(&mut rng);
    let mut used_sequences = HashSet::with_capacity(example_count);

    let files = shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            generate_source_file(
                tokenizer,
                &mut rng,
                split_name,
                index,
                shape,
                &mut used_sequences,
            )
        })
        .collect();
    SourceDataset { files }
}

fn generate_source_file(
    tokenizer: &StrokeTokenizer,
    rng: &mut StdRng,
    split_name: &str,
    index: usize,
    shape: ShapeKind,
    used_sequences: &mut HashSet<Vec<u32>>,
) -> SourceFile {
    let mut content_tokens = shape_tokens(shape, tokenizer, rng);
    for _ in 0..255 {
        if used_sequences.insert(content_tokens.clone()) {
            break;
        }
        content_tokens = shape_tokens(shape, tokenizer, rng);
    }
    let mut tokens = Vec::with_capacity(content_tokens.len() + 2);
    tokens.push(tokenizer.bos_token());
    tokens.extend_from_slice(&content_tokens);
    tokens.push(tokenizer.eot_token());

    SourceFile {
        path: format!("{split_name}/{}-{index:03}.stroke", shape.name()),
        tokens,
        prompt_tokens: content_tokens.clone(),
        target_tokens: content_tokens,
    }
}

fn shape_tokens(shape: ShapeKind, tokenizer: &StrokeTokenizer, rng: &mut StdRng) -> Vec<u32> {
    let mut tokens = Vec::new();
    match shape {
        ShapeKind::Circle => {
            let straight = rng.random_range(1..=2);
            let corner = rng.random_range(1..=2);
            let step = rng.random_range(1..=STEP_LEVELS);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, straight),
                    (Direction::SE, step, corner),
                    (Direction::S, step, straight),
                    (Direction::SW, step, corner),
                    (Direction::W, step, straight),
                    (Direction::NW, step, corner),
                    (Direction::N, step, straight),
                    (Direction::NE, step, corner),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Square => {
            let step = rng.random_range(1..=STEP_LEVELS);
            let side = rng.random_range(1..=5);
            let rotation = rng.random_range(0..Direction::all().len());
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, side),
                    (Direction::S, step, side),
                    (Direction::W, step, side),
                    (Direction::N, step, side),
                ],
                rotation,
                false,
            );
        }
        ShapeKind::Rectangle => {
            let step = rng.random_range(1..=STEP_LEVELS);
            let width = rng.random_range(2..=5);
            let mut height = rng.random_range(1..=4);
            while height == width {
                height = rng.random_range(1..=4);
            }
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, width),
                    (Direction::S, step, height),
                    (Direction::W, step, width),
                    (Direction::N, step, height),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Triangle => {
            let edge = rng.random_range(1..=4);
            let step = rng.random_range(1..=STEP_LEVELS);
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, edge * 2),
                    (Direction::NW, step, edge),
                    (Direction::SW, step, edge),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Diamond => {
            let edge = rng.random_range(1..=5);
            let step = rng.random_range(1..=STEP_LEVELS);
            let rotation = rng.random_range(0..Direction::all().len());
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::NE, step, edge),
                    (Direction::SE, step, edge),
                    (Direction::SW, step, edge),
                    (Direction::NW, step, edge),
                ],
                rotation,
                false,
            );
        }
        ShapeKind::Hexagon => {
            let edge = rng.random_range(1..=3);
            let step = rng.random_range(1..=STEP_LEVELS);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, edge),
                    (Direction::SE, step, edge),
                    (Direction::SW, step, edge),
                    (Direction::W, step, edge),
                    (Direction::NW, step, edge),
                    (Direction::NE, step, edge),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Octagon => {
            let straight = rng.random_range(1..=2);
            let corner = rng.random_range(1..=2);
            let step = rng.random_range(1..=STEP_LEVELS);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, straight),
                    (Direction::SE, step, corner),
                    (Direction::S, step, straight),
                    (Direction::SW, step, corner),
                    (Direction::W, step, straight),
                    (Direction::NW, step, corner),
                    (Direction::N, step, straight),
                    (Direction::NE, step, corner),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::House => {
            let step = rng.random_range(1..=STEP_LEVELS);
            let roof = rng.random_range(1_usize..=2);
            let base = rng.random_range((roof * 2)..=5);
            let wall = rng.random_range(2_usize..=4);
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            let mut segments = vec![
                (Direction::E, step, base),
                (Direction::N, step, wall),
                (Direction::NW, step, roof),
                (Direction::SW, step, roof),
                (Direction::S, step, wall),
            ];
            let floor_return = base.saturating_sub(roof * 2);
            if floor_return > 0 {
                segments.push((Direction::W, step, floor_return));
            }
            trace_moves(&mut tokens, tokenizer, &segments, rotation, mirror);
        }
        ShapeKind::Trapezoid => {
            let step = rng.random_range(1..=STEP_LEVELS);
            let top = rng.random_range(2..=4);
            let slope = rng.random_range(1..=2);
            let bottom = top + slope * 2;
            let rotation = rng.random_range(0..4) * 2;
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, bottom),
                    (Direction::NW, step, slope),
                    (Direction::W, step, top),
                    (Direction::SW, step, slope),
                ],
                rotation,
                mirror,
            );
        }
        ShapeKind::Parallelogram => {
            let step = rng.random_range(1..=STEP_LEVELS);
            let width = rng.random_range(2..=5);
            let lean = rng.random_range(1..=2);
            let rotation = rng.random_range(0..Direction::all().len());
            let mirror = rng.random_bool(0.5);
            trace_moves(
                &mut tokens,
                tokenizer,
                &[
                    (Direction::E, step, width),
                    (Direction::NE, step, lean),
                    (Direction::W, step, width),
                    (Direction::SW, step, lean),
                ],
                rotation,
                mirror,
            );
        }
    }
    if shape.is_closed() {
        apply_closed_loop_variant(tokenizer, rng, &mut tokens);
    }
    tokens
}

#[cfg(test)]
fn token_displacement(tokenizer: &StrokeTokenizer, tokens: &[u32]) -> (i32, i32) {
    let mut x = 0;
    let mut y = 0;
    for &token in tokens {
        if let TokenKind::Move(direction, step) = tokenizer.decode_token_kind(token) {
            let (dx, dy) = direction.delta(step);
            x += dx;
            y += dy;
        }
    }
    (x, y)
}

fn transformed_direction(
    direction: Direction,
    rotation: usize,
    mirror_horizontal: bool,
) -> Direction {
    let direction = if mirror_horizontal {
        direction.mirror_horizontal()
    } else {
        direction
    };
    direction.rotate(rotation)
}

fn repeat_move(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    direction: Direction,
    step: u8,
    count: usize,
) {
    for _ in 0..count {
        tokens.push(tokenizer.move_token(direction, step));
    }
}

fn trace_moves(
    tokens: &mut Vec<u32>,
    tokenizer: &StrokeTokenizer,
    segments: &[(Direction, u8, usize)],
    rotation: usize,
    mirror_horizontal: bool,
) {
    for &(direction, step, count) in segments {
        repeat_move(
            tokens,
            tokenizer,
            transformed_direction(direction, rotation, mirror_horizontal),
            step,
            count,
        );
    }
}

fn apply_closed_loop_variant(tokenizer: &StrokeTokenizer, rng: &mut StdRng, tokens: &mut Vec<u32>) {
    if tokens.is_empty() {
        return;
    }

    let start_offset = rng.random_range(0..tokens.len());
    tokens.rotate_left(start_offset);

    if rng.random_bool(0.5) {
        let reversed = tokens
            .iter()
            .rev()
            .map(|&token| match tokenizer.decode_token_kind(token) {
                TokenKind::Move(direction, step) => {
                    tokenizer.move_token(direction.opposite(), step)
                }
                _ => token,
            })
            .collect();
        *tokens = reversed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_gguf::{GgufMetadata, GgufVersion};
    use std::{collections::HashSet, io::Cursor};

    #[test]
    fn tokenizer_survives_gguf_metadata_round_trip() {
        let tokenizer = StrokeTokenizer::new();
        let gguf = GgufMetadata {
            version: GgufVersion::V3,
            metadata: tokenizer
                .gguf_metadata()
                .into_iter()
                .map(|(key, value)| (key.into_boxed_str(), value))
                .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let mut bytes = Cursor::new(Vec::new());
        gguf.write(&mut bytes, std::iter::empty::<(&str, &[u8])>())
            .unwrap();
        let mut reader = Cursor::new(bytes.into_inner());
        let vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let restored = StrokeTokenizer::from_var_builder(&vb).unwrap();
        assert_eq!(restored.vocab_size(), tokenizer.vocab_size());
        assert_eq!(restored.bos_token(), tokenizer.bos_token());
        assert_eq!(restored.eot_token(), tokenizer.eot_token());
    }

    #[test]
    fn synthetic_dataset_split_uses_requested_sizes() {
        let tokenizer = StrokeTokenizer::new();
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 24,
            batch_size: 4,
            n_embd: 32,
            n_head: 2,
            n_ff: 64,
            n_layer: 2,
            conv_kernel_size: 3,
            attention_period: 1,
            eps: 1e-5,
            init_scale: 0.08,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: crate::config::SaveQuantization::F32,
            train_examples: 7,
            validation_examples: 3,
            test_examples: 2,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let split = synthetic_dataset_split(&tokenizer, &config);
        assert_eq!(split.train.num_docs(), 7);
        assert_eq!(split.validation.num_docs(), 3);
        assert_eq!(split.test.num_docs(), 2);
    }

    #[test]
    fn synthetic_dataset_stays_balanced_and_within_context_window() {
        let tokenizer = StrokeTokenizer::new();
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 24,
            batch_size: 8,
            n_embd: 24,
            n_head: 2,
            n_ff: 48,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            eps: 1e-5,
            init_scale: 0.1,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: crate::config::SaveQuantization::F32,
            train_examples: ShapeKind::all().len() * 2,
            validation_examples: ShapeKind::all().len(),
            test_examples: ShapeKind::all().len(),
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let split = synthetic_dataset_split(&tokenizer, &config);
        let seen_shapes = split
            .train
            .files()
            .iter()
            .map(|file| {
                let filename = file.path().split('/').next_back().unwrap();
                filename
                    .rsplit_once('-')
                    .map(|(shape, _)| shape.to_string())
                    .unwrap()
            })
            .collect::<HashSet<_>>();

        assert_eq!(seen_shapes.len(), ShapeKind::all().len());
        let unique_token_sequences = split
            .train
            .files()
            .iter()
            .map(|file| file.target_tokens().to_vec())
            .collect::<HashSet<_>>();
        assert_eq!(unique_token_sequences.len(), split.train.files().len());
        assert!(
            split
                .train
                .files()
                .iter()
                .all(|file| file.tokens.len() <= config.block_size + 1)
        );
    }

    #[test]
    fn synthetic_shapes_are_closed_loops() {
        let tokenizer = StrokeTokenizer::new();
        for shape in ShapeKind::all() {
            for seed in 0..32 {
                let mut rng = StdRng::seed_from_u64(seed);
                let tokens = shape_tokens(shape, &tokenizer, &mut rng);
                assert_eq!(
                    token_displacement(&tokenizer, &tokens),
                    (0, 0),
                    "{} should close back to the origin",
                    shape.name()
                );
            }
        }
    }

    #[test]
    fn closed_loop_variants_shift_and_reverse_sequences() {
        let tokenizer = StrokeTokenizer::new();
        let base = vec![
            tokenizer.move_token(Direction::E, 1),
            tokenizer.move_token(Direction::E, 1),
            tokenizer.move_token(Direction::S, 1),
            tokenizer.move_token(Direction::S, 1),
            tokenizer.move_token(Direction::W, 1),
            tokenizer.move_token(Direction::W, 1),
            tokenizer.move_token(Direction::N, 1),
            tokenizer.move_token(Direction::N, 1),
        ];

        let variants = (0..32)
            .map(|seed| {
                let mut tokens = base.clone();
                apply_closed_loop_variant(
                    &tokenizer,
                    &mut StdRng::seed_from_u64(seed),
                    &mut tokens,
                );
                assert_eq!(token_displacement(&tokenizer, &tokens), (0, 0));
                tokens
            })
            .collect::<HashSet<_>>();

        assert!(
            variants.len() > 4,
            "expected shifted/reversed closed-loop variants"
        );
    }

    #[test]
    fn sample_export_produces_valid_svg_bytes() {
        let tokenizer = StrokeTokenizer::new();
        let tokens = vec![
            tokenizer.move_token(Direction::E, 2),
            tokenizer.move_token(Direction::SE, 1),
            tokenizer.move_token(Direction::S, 2),
        ];
        let root = std::env::temp_dir().join("nanochat-stroke-sample.svg");
        write_tokens_to_svg_file(&tokenizer, &tokens[..2], &tokens[2..], &root);
        let svg = fs::read_to_string(&root).unwrap();
        assert!(svg.starts_with("<svg"));
        let _ = fs::remove_file(root);
    }
}
