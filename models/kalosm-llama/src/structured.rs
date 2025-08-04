use cfg::{DenseGrammar, Recognizer};
use kalosm_sample::CreateParserState;
use kalosm_sample::ParseStatus;
use llm_samplers::prelude::{Logit, Logits};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::{
    fmt::{Debug, Display, Formatter},
    sync::{Arc, Mutex},
};
use tokenizers::tokenizer::Tokenizer;

use crate::model::LlamaModelError;
use crate::token_stream::TokenOutputStream;
use crate::{LlamaModel, LlamaSession};

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_structured<P>(
    prompt: impl Display,
    llm: &LlamaModel,
    session: &mut LlamaSession,
    cfg: Option<&DenseGrammar>,
    parser: &P,
    mut sampler: Arc<Mutex<dyn Sampler>>,
    mut on_token: impl FnMut(String, u32) -> Result<(), LlamaModelError>,
    top_k: Option<usize>,
    seed: Option<u64>,
    trie: &mut EvaluationTrie,
) -> Result<P::Output, LlamaModelError>
where
    P: CreateParserState + 'static,
{
    let eos_token = llm.model.config.stop_token_string.clone();
    let mut on_token = move |tok: String, token_id: u32| {
        if tok == eos_token {
            return Ok(());
        }
        on_token(tok, token_id)
    };
    let mut session = session
        .cache
        .write()
        .map_err(|err| LlamaModelError::Session(err.to_string()))?;
    let tokenizer = &llm.tokenizer;

    let prompt_text = prompt.to_string();
    let prompt_tokens = tokenizer
        .encode_fast(prompt_text, false)
        .map_err(LlamaModelError::Tokenizer)?;
    let mut prompt_tokens = prompt_tokens.get_ids();

    // Prompt healing
    // Trim the last token and add what it would decode to into the constraints
    let last_token = if let Some((last, tokens)) = prompt_tokens.split_last() {
        if tokenizer.get_added_tokens_decoder().contains_key(last) {
            None
        } else {
            prompt_tokens = tokens;
            Some(*last)
        }
    } else {
        None
    };

    let mut unprocessed_token_count = prompt_tokens.len();
    let mut token_stream = TokenOutputStream::new(tokenizer.clone());
    for token in prompt_tokens {
        token_stream
            .next_token(*token)
            .map_err(LlamaModelError::TokenOutputStreamError)?;
    }

    let remaining_prompt_text = last_token
        .map(|token| {
            token_stream
                .peek_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)
        })
        .transpose()?
        .flatten()
        .unwrap_or_default();

    let mut parser_state = cfg.map(Recognizer::new);
    let mut extra_parser_state = parser.create_parser_state();
    let mut strip_required_next = true;

    let mut rng = if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    let mut state_map = vec![];
    let mut logits_indexed = Logits::default();
    let mut token_cache = DetokenizationCache::new();
    let mut logits = Logits::default();
    let mut logit_probs = Vec::new();

    let mut current_token = None;

    loop {
        let tokens = token_stream.tokens();
        LlamaModel::forward(
            &llm.model,
            &llm.device,
            &tokens[tokens.len() - unprocessed_token_count..],
            Some(&mut *session),
            &mut logit_probs,
            &llm.tokenizer,
        )?;
        let resources = &mut SamplerResources {
            previous_tokens: tokens,
            rng: &mut rng,
        };

        // fill the state map with None for each token
        token_cache.clear(logit_probs.len());
        state_map.clear();
        logits_indexed.clear();
        logits.clear();
        for (id, prob) in logit_probs.iter().enumerate() {
            logits_indexed.push(Logit {
                token_id: id as u32,
                logit: *prob,
                prob: 0f32,
            });
            state_map.push(None);
        }
        logits_indexed.set_softmax(false);
        logits_indexed.ensure_softmax().unwrap();

        let mut valid_tokens = false;

        // If we don't have a top k, then we can just cache the entire detokenization
        if top_k.is_none() {
            token_cache.expand(
                &(0..logit_probs.len() as u32).collect::<Vec<_>>(),
                &token_stream,
            );
        }

        const DETOKENIZATION_INITIAL_BATCH_SIZE: usize = 64;

        // Constraints tend to be either very difficult to satisfy or very easy to satisfy
        // We exponentially increase the batch size as a balance between the two
        // If the first half of the tokens are invalid, it is unlikely that the first 64 tokens of the second half will be valid
        let mut detokenization_batch_size = DETOKENIZATION_INITIAL_BATCH_SIZE;

        let mut partitioned_logits_index = top_k.map(|_| 0);

        for i in 0..logits_indexed.len() {
            // If we have top k enabled, and there are less than top k - committed logits sorted, we need to expand the partitioned logits
            if let (Some(top_k), Some(partitioned_index)) = (top_k, partitioned_logits_index) {
                // If the remaining logits are less than the top k, no need to partition
                let remaining_needed = top_k - logits.len();
                let remaining_possible = partitioned_index - i;
                if remaining_possible <= remaining_needed {
                    // We batch together updates to the cache by detokenization_batch_size
                    let logits_to_update = remaining_needed
                        .max(detokenization_batch_size)
                        .min(logits_indexed.len() - 1 - i);
                    let new_partitioned_index = i + logits_to_update;

                    // If we eliminated a logit, our partitioning of the logits is no longer valid
                    logits_indexed[i..].select_nth_unstable_by(logits_to_update, cmp_logits);
                    logits_indexed[i..=new_partitioned_index].sort_unstable_by(cmp_logits);
                    // Expand the cache to include the new logits
                    partitioned_logits_index = Some(new_partitioned_index);
                    token_cache.expand_with_logits(
                        &logits_indexed[i..=new_partitioned_index],
                        &token_stream,
                    );

                    // Double the batch size for next time
                    detokenization_batch_size = detokenization_batch_size.saturating_mul(4);
                }
            }

            let Logit {
                token_id,
                logit,
                prob,
                ..
            } = logits_indexed[i];

            let Some(text) = token_cache.get(token_id as usize) else {
                continue;
            };
            let state_after_push = parser_state.as_mut().map(|parser_state| parser_state.push(token_id));
            let could_become_valid = state_after_push.as_ref().map_or(true, |state_after_push| {
                state_after_push.could_become_valid()
            });

            if could_become_valid {
                let result = parser.parse(&extra_parser_state, text.as_bytes());
                trie.push(token_id, prob as f64, current_token, result.is_ok(), false);
                if let Ok(result) = result {
                    // println!(
                    //     "Token {:?} with probability {} could become valid",
                    //     tokenizer.id_to_token(token_id),
                    //     prob
                    // );

                    let parsed_bytes = match &result {
                        ParseStatus::Finished { remaining, .. } => text.len() - remaining.len(),
                        ParseStatus::Incomplete { .. } => text.len(),
                    };
                    let result = result.without_remaining();
                    state_map[token_id as usize] =
                        Some((state_after_push, parser_state.clone(), result, parsed_bytes));
                    valid_tokens = true;
                    logits.push(Logit {
                        token_id,
                        logit,
                        prob: 0f32,
                    });
                    // If we only need to keep the top k logits, then we can quit early once we have enough
                    if let Some(top_k) = top_k {
                        if logits.len() >= top_k {
                            break;
                        }
                    }
                }
            } else {
                // trie.push(
                //     token_id,
                //     prob as f64,
                //     current_token,
                //     false,
                //     false,
                // );
            }
            if let Some(parser_state) = &mut parser_state {
                parser_state.pop();
            }
        }

        // If there are no valid tokens, return an error
        if !valid_tokens {
            return Err(LlamaModelError::NoValidTokens);
        }
        let token_id = {
            // softmax logits
            logits.set_softmax(false);
            logits.set_sorted(false);
            logits.ensure_softmax().unwrap();
            logits.ensure_sorted().unwrap();
            for logit in &mut *logits {
                let estimate = match current_token {
                    Some(current) => trie.nodes[current]
                        .evaluated_children
                        .get(&logit.token_id)
                        .map(|i| trie.estimated_probability(*i)),
                    None => trie
                        .roots
                        .get(&logit.token_id)
                        .map(|i| trie.estimated_probability(*i)),
                };
                if let Some(estimate) = estimate {
                    logit.logit = estimate as f32;
                    logit.prob = estimate as f32;
                } else {
                    logit.logit = logit.prob.exp();
                }
            }
            // println!(
            //     "Sum of logits: {}",
            //     logits.iter().map(|logit| logit.logit).sum::<f32>()
            // );
            // println!(
            //     "Sum of probabilities: {}",
            //     logits.iter().map(|logit| logit.prob).sum::<f32>()
            // );
            logits.retain(|logit| logit.prob > 0.0);
            logits.set_softmax(false);
            logits.ensure_softmax().unwrap();

            let mut sampled_logits = logits.clone();
            let token_id = sampler
                .sample_token(resources, &mut sampled_logits)
                .map_err(|err| LlamaModelError::SamplerError(err.into()))?
                .ok_or_else(|| {
                    LlamaModelError::SamplerError(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "Sampler returned None. Failed to sample token from logits: {:?}",
                            sampled_logits
                        ),
                    )))
                })?;

            token_id
        };

        current_token = Some(match current_token {
            Some(current) => *trie.nodes[current]
                .evaluated_children
                .get(&token_id)
                .unwrap(),
            None => *trie.roots.get(&token_id).unwrap(),
        });
        // println!(
        //     "\nsampled current_token raw: {:?} with prob {}",
        //     current_token,
        //     logits
        //         .iter()
        //         .find(|logit| logit.token_id == token_id)
        //         .unwrap()
        //         .prob
        // );
        // println!(
        //     "current_token: {:?}\n",
        //     current_token.map(|id| tokenizer.decode(&[trie.nodes[id].token], false))
        // );

        // If this and the last token would result in a valid merge, then the probability in the training data should be close
        // to zero
        if let Some(&last) = token_stream.tokens().last() {
            let pair = [last, token_id];
            if llm.merges.contains(&pair) {
                eprintln!(
                    "ERROR: Tokens {:?} and {:?} should have already merged into {:?}",
                    tokenizer.id_to_token(last),
                    tokenizer.id_to_token(token_id),
                    tokenizer.decode(&pair, false)
                );
            }
        }

        unprocessed_token_count = 1;
        let (_, mut result, mut extra_state, _) = state_map
            .get_mut(token_id as usize)
            .unwrap()
            .take()
            .unwrap_or_else(|| panic!("Token {} not found in state map", token_id));
        let mut token = token_stream
            .next_token(token_id)
            .map_err(LlamaModelError::TokenOutputStreamError)?
            .unwrap();
        // If we are still loading the initial prompt, don't send that part of the text
        if strip_required_next {
            if let Some(stripped) = token.strip_prefix(&remaining_prompt_text) {
                token = stripped.to_string();
            }
            strip_required_next = false;
        }
        on_token(token, token_id)?;

        if let Some(result) = result.as_mut() {
            let mut possible_next = result.possible_next_terminals();
            while possible_next.len() == 1 {
                let token_id = *possible_next.iter().next().unwrap();
                let mut token = token_stream
                    .next_token(token_id)
                    .map_err(LlamaModelError::TokenOutputStreamError)?
                    .unwrap();
                unprocessed_token_count += 1;
                // If we are still loading the initial prompt, don't send that part of the text
                if strip_required_next {
                    if let Some(stripped) = token.strip_prefix(&remaining_prompt_text) {
                        token = stripped.to_string();
                    }
                    strip_required_next = false;
                }
                // println!("skipping forward with token: {}", token);
                on_token(token, token_id)?;
                result.push(token_id);
                possible_next = result.possible_next_terminals();
                let as_str = tokenizer
                    .decode(&[token_id], false)
                    .map_err(LlamaModelError::Tokenizer)?;
                let (new_state, _) = extra_state.unwrap_incomplete();
                let result = parser.parse(&new_state, as_str.as_bytes());
                // add the token to the trie
                current_token =
                    Some(trie.push(token_id as u32, 1.0, current_token, result.is_ok(), true));
                match result {
                    Ok(new_state) => extra_state = new_state.without_remaining(),
                    Err(_) => {
                        return Err(LlamaModelError::SamplerError(Box::new(
                            std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("No valid next token for {:?}", as_str,),
                            ),
                        )))
                    }
                }
            }
        }

        parser_state = result.clone();
        match extra_state {
            ParseStatus::Finished { result, .. } => {
                return Ok(result);
            }
            ParseStatus::Incomplete { new_state, .. } => {
                extra_parser_state = new_state.clone();
            }
        }
    }
}

#[derive(Debug)]
pub struct EvaluationTrie {
    roots: FxHashMap<u32, usize>,
    nodes: Vec<EvaluationNode>,
}

impl Default for EvaluationTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationTrie {
    pub fn new() -> Self {
        println!("Creating new EvaluationTrie");
        Self {
            roots: Default::default(),
            nodes: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        println!("Clearing EvaluationTrie");
        self.roots.clear();
        self.nodes.clear();
    }

    // Make a sequence impossible
    pub fn make_sequence_impossible(&mut self, tokens: &[u32]) {
        let mut position = None;
        for &token in tokens {
            match position {
                Some(pos) => {
                    let node: &mut EvaluationNode = &mut self.nodes[pos];
                    if let Some(&child_id) = node.evaluated_children.get(&token) {
                        position = Some(child_id);
                    } else {
                        position = None;
                    }
                }
                None => {
                    if let Some(&child_id) = self.roots.get(&token) {
                        position = Some(child_id);
                    } else {
                        position = None;
                    }
                }
            }
        }

        if let Some(pos) = position {
            let node = &mut self.nodes[pos];
            node.in_grammar = false;
            node.probability = 0.0;
            node.from_tokenization_constraint = true;
        }
    }

    fn push(
        &mut self,
        token: u32,
        probability: f64,
        parent: Option<usize>,
        in_grammar: bool,
        from_tokenization_constraint: bool,
    ) -> usize {
        let id = match parent {
            Some(parent_id) => {
                let parent = &self.nodes[parent_id];
                if let Some(id) = parent.evaluated_children.get(&token).copied() {
                    let node = &mut self.nodes[id];
                    node.token = token;
                    node.probability = node.probability.min(probability);
                    node.in_grammar &= in_grammar;
                    node.from_tokenization_constraint |= from_tokenization_constraint;
                    return id;
                }
                let id =
                    self.create_node(token, probability, in_grammar, from_tokenization_constraint);
                self.nodes[parent_id].evaluated_children.insert(token, id);
                id
            }
            None => {
                if let Some(id) = self.roots.get(&token).copied() {
                    let node = &mut self.nodes[id];
                    node.token = token;
                    node.probability = node.probability.min(probability);
                    node.in_grammar &= in_grammar;
                    node.from_tokenization_constraint |= from_tokenization_constraint;
                    return id;
                }
                let id =
                    self.create_node(token, probability, in_grammar, from_tokenization_constraint);
                self.roots.insert(token, id);
                id
            }
        };
        debug_assert!(!self.check_for_cycle(id, &HashSet::new()));
        id
    }

    fn check_for_cycle(&self, node: usize, parents: &HashSet<usize>) -> bool {
        if parents.contains(&node) {
            return true;
        }
        let mut new_parents = parents.clone();
        new_parents.insert(node);
        let node = &self.nodes[node];
        for child in node.evaluated_children.values() {
            if self.check_for_cycle(*child, &new_parents) {
                return true;
            }
        }
        false
    }

    fn create_node(
        &mut self,
        token: u32,
        probability: f64,
        in_grammar: bool,
        from_tokenization_constraint: bool,
    ) -> usize {
        let node = EvaluationNode {
            token,
            probability,
            in_grammar,
            from_tokenization_constraint,
            evaluated_children: Default::default(),
        };
        if probability <= -0.1 || probability > 1.1 {
            tracing::error!(
                "Probability of token {} is {}, this should never happen",
                token,
                probability
            );
        }
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    fn estimated_probability(&self, node: usize) -> f64 {
        let node = &self.nodes[node];
        if !node.in_grammar {
            return 0.0;
        }
        let initial_estimate = node.probability;
        if node.evaluated_children.is_empty() {
            return initial_estimate;
        }
        let sum_probability_of_children = node
            .evaluated_children
            .values()
            .map(|child| self.estimated_probability(*child))
            .sum::<f64>();
        let result = initial_estimate * sum_probability_of_children;
        assert!((0.0..=1.0).contains(&result));
        result
    }

    pub fn graphvis(&self, tokenizer: &Tokenizer) -> String {
        let mut graph = String::new();
        graph.push_str("digraph G {\n");
        fn filter_children(
            trie: &EvaluationTrie,
            children_slice: &FxHashMap<u32, usize>,
        ) -> Vec<usize> {
            let mut children = Vec::new();
            let mut included = Vec::new();
            for child_id in children_slice.values() {
                let child = &trie.nodes[*child_id];
                if !child.evaluated_children.is_empty() {
                    included.push(*child_id);
                } else {
                    children.push(*child_id);
                }
            }

            children.sort_by(|a, b| {
                let a = &trie.nodes[*a];
                let b = &trie.nodes[*b];

                if a.probability < b.probability {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });
            included.extend(children.iter().take(4).copied());
            included
        }
        let mut queue = filter_children(self, &self.roots);
        while let Some(node_id) = queue.pop() {
            let start_id = node_id;
            let mut node_id = node_id;
            let mut all_text = String::new();
            loop {
                let node = &self.nodes[node_id];
                all_text += &tokenizer
                    .id_to_token(node.token)
                    .unwrap()
                    .escape_debug()
                    .to_string();
                let probable_children = filter_children(self, &node.evaluated_children);

                match &*probable_children {
                    [] => break,
                    [single_item] => {
                        node_id = *single_item;
                    }
                    _ => {
                        for child_id in &probable_children {
                            graph.push_str(&format!("  {} -> {};\n", start_id, child_id));
                        }
                        queue.extend(probable_children);
                        break;
                    }
                }
            }
            let start_node = &self.nodes[start_id];
            graph.push_str(&format!(
                "  {} [label=\"{}; prob {}; updated {}\" {} {}];\n",
                start_id,
                all_text,
                start_node.probability,
                self.estimated_probability(start_id),
                if start_node.evaluated_children.is_empty() {
                    if start_node.in_grammar {
                        "color=black"
                    } else {
                        "color=red"
                    }
                } else {
                    "color=green"
                },
                if start_node.from_tokenization_constraint {
                    " style=dashed"
                } else {
                    ""
                }
            ));
            if start_id != node_id {
                graph.push_str(&format!("  {} -> {};\n", start_id, node_id));
            }
        }
        graph.push_str("}\n");
        graph
    }

    /// Calculate the Shannon entropy of the tree
    pub fn shannon_entropy(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut current_nodes = self
            .roots
            .values()
            .map(|index| (self.estimated_probability(*index), *index))
            .collect::<Vec<_>>();
        let total_root_prob = current_nodes.iter().map(|(prob, _)| prob).sum::<f64>();
        for (prob, _) in &mut current_nodes {
            if *prob > 0.0 {
                *prob /= total_root_prob;
            }
        }

        let mut leaf_node_probabilities = Vec::new();
        while let Some((parent_prob, node_id)) = current_nodes.pop() {
            let node = &self.nodes[node_id];
            if !node.evaluated_children.is_empty() {
                let child_probabilities = node
                    .evaluated_children
                    .values()
                    .map(|child| (self.estimated_probability(*child), *child))
                    .collect::<Vec<_>>();
                let total_child_prob = child_probabilities
                    .iter()
                    .map(|(prob, _)| prob)
                    .sum::<f64>();
                current_nodes.extend(child_probabilities.into_iter().map(|(prob, child_id)| {
                    let prob = if prob == 0.0 {
                        prob
                    } else {
                        (prob / total_child_prob) * parent_prob
                    };
                    (prob, child_id)
                }));
            } else {
                leaf_node_probabilities.push(parent_prob);
            }
        }
        let total_probability: f64 = leaf_node_probabilities.iter().copied().sum();
        assert!(0.9 < total_probability && total_probability < 1.1);
        let entropy: f64 = leaf_node_probabilities
            .iter()
            .map(|&prob| {
                let prob = prob / total_probability;
                if prob > 0.0 {
                    -prob * prob.log2()
                } else {
                    0.0
                }
            })
            .sum();
        entropy
    }
}

#[test]
fn test_evaluation_trie() {
    let mut trie = EvaluationTrie::new();
    let node_1 = trie.push(1, 1.0, None, true, false);
    let node_2 = trie.push(2, 0.5, Some(node_1), true, false);
    let node_3 = trie.push(3, 0.5, Some(node_1), true, false);
    let node_4 = trie.push(4, 0.2, Some(node_2), false, false);
    let node_5 = trie.push(5, 0.8, Some(node_2), true, false);
    let node_6 = trie.push(6, 1.0, Some(node_3), false, false);

    assert_eq!(trie.estimated_probability(node_1), 0.4);
    assert_eq!(trie.estimated_probability(node_2), 0.4);
    assert_eq!(trie.estimated_probability(node_3), 0.0);
    assert_eq!(trie.estimated_probability(node_4), 0.0);
    assert_eq!(trie.estimated_probability(node_5), 0.8);
    assert_eq!(trie.estimated_probability(node_6), 0.0);
}

#[derive(Debug)]
struct EvaluationNode {
    token: u32,
    probability: f64,
    in_grammar: bool,
    from_tokenization_constraint: bool,
    evaluated_children: FxHashMap<u32, usize>,
}

fn cmp_logits(a: &Logit, b: &Logit) -> std::cmp::Ordering {
    f32::total_cmp(&b.logit, &a.logit)
}

struct SamplerResources<'a, 'b, R: rand::Rng> {
    rng: &'a mut R,
    previous_tokens: &'b [u32],
}

impl<R> Debug for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("previous_tokens", &self.previous_tokens)
            .finish()
    }
}

impl<R> HasSamplerResources for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        fun(self.rng);
        Ok(())
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[u32])) -> Result<(), SamplerError> {
        fun(self.previous_tokens);
        Ok(())
    }
}

#[derive(Clone)]
enum TokenCacheStatus {
    Empty,
    Invalid,
    Valid(String),
}

struct DetokenizationCache {
    cache: Box<[TokenCacheStatus]>,
    vec: Vec<Option<String>>,
}

impl DetokenizationCache {
    fn new() -> Self {
        Self {
            cache: Box::new([]),
            vec: Vec::new(),
        }
    }

    fn get(&self, index: usize) -> Option<&str> {
        match &self.cache[index] {
            TokenCacheStatus::Empty => panic!("cache for token {} is empty", index),
            TokenCacheStatus::Invalid => None,
            TokenCacheStatus::Valid(token) => Some(token),
        }
    }

    fn expand_with_logits(&mut self, tokens: &[Logit], stream: &TokenOutputStream) {
        stream.peek_tokens(
            tokens.into_par_iter().map(|logit| logit.token_id),
            &mut self.vec,
        );

        for (logit, token) in tokens.iter().zip(self.vec.drain(..)) {
            self.cache[logit.token_id as usize] = match token {
                Some(token) => TokenCacheStatus::Valid(token),
                None => TokenCacheStatus::Invalid,
            };
        }
    }

    fn expand(&mut self, tokens: &[u32], stream: &TokenOutputStream) {
        stream.peek_tokens(tokens.into_par_iter().copied(), &mut self.vec);

        for (&i, token) in tokens.iter().zip(self.vec.drain(..)) {
            self.cache[i as usize] = match token {
                Some(token) => TokenCacheStatus::Valid(token),
                None => TokenCacheStatus::Invalid,
            };
        }
    }

    fn clear(&mut self, size: usize) {
        if self.cache.len() == size {
            for token in self.cache.iter_mut() {
                *token = TokenCacheStatus::Empty;
            }
        } else {
            self.cache = vec![TokenCacheStatus::Empty; size].into_boxed_slice();
        }
        self.vec.clear();
    }
}
