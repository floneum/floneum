use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ArcLinkedList<T> {
    pub len: usize,
    pub tail: Option<ArcLinkedListNode<T>>,
}

impl<T> ArcLinkedList<T> {
    pub(crate) fn push(&mut self, value: Arc<T>) {
        self.len += 1;
        match self.tail.take() {
            Some(tail) => {
                let node = ArcLinkedListNode {
                    prev: Some(Arc::new(tail)),
                    value,
                };
                self.tail = Some(node);
            }
            None => {
                self.tail = Some(ArcLinkedListNode { prev: None, value });
            }
        }
    }

    /// Get the length of the list.
    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut vec = Vec::with_capacity(self.len);
        if let Some(mut node) = self.tail.as_ref() {
            vec.push((*node.value).clone());
            while let Some(prev) = node.prev.as_ref() {
                vec.push((*prev.value).clone());
                node = prev;
            }
            vec.reverse();
        }
        vec
    }
}

impl<T> Default for ArcLinkedList<T> {
    fn default() -> Self {
        Self { len: 0, tail: None }
    }
}

impl<T> Clone for ArcLinkedList<T> {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            tail: self.tail.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ArcLinkedListNode<T> {
    pub(crate) prev: Option<Arc<ArcLinkedListNode<T>>>,
    pub(crate) value: Arc<T>,
}

impl<T> Clone for ArcLinkedListNode<T> {
    fn clone(&self) -> Self {
        Self {
            prev: self.prev.clone(),
            value: self.value.clone(),
        }
    }
}
