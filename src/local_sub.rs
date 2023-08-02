use dioxus::prelude::*;
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
    sync::Arc,
};

pub struct UseLocalSubscription<T> {
    scope_id: ScopeId,
    inner: LocalSubscription<T>,
}

impl<T> Drop for UseLocalSubscription<T> {
    fn drop(&mut self) {
        self.inner
            .subscriptions
            .borrow_mut()
            .retain(|(id, _)| *id != self.scope_id);
    }
}

impl<T> UseLocalSubscription<T> {
    pub fn read(&self) -> Ref<T> {
        self.inner.inner.borrow()
    }

    pub fn write(&self) -> RefMut<T> {
        for (_, sub) in self.inner.subscriptions.borrow().iter() {
            sub();
        }
        self.inner.inner.borrow_mut()
    }
}

pub struct LocalSubscription<T> {
    inner: Rc<RefCell<T>>,
    #[allow(clippy::type_complexity)]
    subscriptions: Rc<RefCell<Vec<(ScopeId, Arc<dyn Fn()>)>>>,
}

impl<T> Clone for LocalSubscription<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            subscriptions: self.subscriptions.clone(),
        }
    }
}

impl<T> PartialEq for LocalSubscription<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: 'static> LocalSubscription<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
            subscriptions: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn use_<'a>(&self, cx: &'a ScopeState) -> &'a UseLocalSubscription<T> {
        cx.use_hook(move || {
            let myself: LocalSubscription<T> = self.clone();
            let scope_id = cx.scope_id();
            myself
                .subscriptions
                .borrow_mut()
                .push((scope_id, cx.schedule_update()));
            UseLocalSubscription {
                inner: myself,
                scope_id,
            }
        })
    }

    pub fn write(&self) -> RefMut<T> {
        for (_, sub) in self.subscriptions.borrow().iter() {
            sub();
        }
        self.inner.borrow_mut()
    }

    pub fn read(&self, _: &ScopeState) -> Ref<T> {
        self.inner.borrow()
    }

    pub fn read_silent(&self) -> Ref<T> {
        self.inner.borrow()
    }
}
