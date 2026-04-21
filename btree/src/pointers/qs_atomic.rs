use thin::{Pointable, QsOwned, QsShared};

use super::OwnedThinAtomicPtr;
use crate::sync::Ordering;

/// Lock-free atomic cell that reclaims replaced values via QSBR.
/// Writers are single-writer and must serialize themselves.
pub struct QsAtomic<T: Pointable> {
    slot: OwnedThinAtomicPtr<T>,
}

impl<T: Pointable> QsAtomic<T> {
    pub fn new(value: T) -> Self {
        Self {
            slot: OwnedThinAtomicPtr::new(QsOwned::new(value)),
        }
    }

    #[inline]
    pub fn load(&self) -> QsShared<T> {
        self.slot
            .load_shared(Ordering::Acquire)
            .expect("QsAtomic is always populated")
    }

    /// Single-writer only: concurrent calls from multiple threads race the
    /// load/swap and can leak or double-publish. This is load/compute/swap,
    /// not a CAS loop.
    #[inline]
    pub fn update_with(&self, f: impl FnOnce(&T) -> T) {
        let next = f(&self.load());
        let _old = self.slot.swap(QsOwned::new(next), Ordering::AcqRel);
    }
}

impl<T: Pointable> Drop for QsAtomic<T> {
    fn drop(&mut self) {
        // SAFETY: `&mut self` rules out concurrent writers; we take ownership
        // exactly once and let the resulting `QsOwned` schedule QSBR reclamation.
        let _owned = unsafe {
            self.slot
                .load_owned(Ordering::Acquire)
                .expect("QsAtomic is always populated")
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use btree_macros::qsbr_test;

    #[qsbr_test]
    fn test_load_returns_initial_value() {
        let atom = QsAtomic::new(42usize);
        assert_eq!(*atom.load(), 42);
    }

    #[qsbr_test]
    fn test_update_with_publishes_new_value() {
        let atom = QsAtomic::new(10usize);

        atom.update_with(|current| current + 1);
        assert_eq!(*atom.load(), 11);

        atom.update_with(|current| current * 2);
        assert_eq!(*atom.load(), 22);
    }

    #[qsbr_test]
    fn test_prior_loads_stay_valid_across_update() {
        let atom = QsAtomic::new(100usize);
        let before = atom.load();

        atom.update_with(|_| 200);
        let after = atom.load();

        assert_eq!(*before, 100);
        assert_eq!(*after, 200);
    }

    #[qsbr_test]
    fn test_drop_does_not_leak_or_panic() {
        let atom = QsAtomic::new(vec![1u32, 2, 3]);
        assert_eq!(atom.load().len(), 3);
        drop(atom);
    }
}
