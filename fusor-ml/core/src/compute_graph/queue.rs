use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use super::AnyComputeKey;

#[derive(Default, Debug)]
pub(crate) struct ComputeQueue {
    nodes: VecDeque<AnyComputeKey>,
    set: FxHashSet<AnyComputeKey>,
}

impl ComputeQueue {
    pub(crate) fn push_back(&mut self, key: AnyComputeKey) {
        if self.set.insert(key) {
            self.nodes.push_back(key);
        }
    }

    pub(crate) fn pop_front(&mut self) -> Option<AnyComputeKey> {
        self.nodes.pop_front().inspect(|key| {
            self.set.remove(key);
        })
    }
}
