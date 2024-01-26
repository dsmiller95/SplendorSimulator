use crate::wrapper_test::core_obj::{CoreObj, CoreTrait};

pub struct IndexScopedCoreObj<'a, T> {
    pub wrapped: &'a mut CoreObj<T>,
    pub index: usize,
}

pub trait IndexScopedCoreTrait<T> {
    fn get_some_number(&self) -> i32;
    fn set_some_number(&mut self, some_number: i32);

    fn get_some_item(&self) -> &Option<T>;
    fn set_some_item(&mut self, some_item: Option<T>);
}

impl <T> IndexScopedCoreObj<'_, T> {
    pub fn new(wrapped: &mut CoreObj<T>, index: usize) -> IndexScopedCoreObj<T> {
        IndexScopedCoreObj {
            wrapped,
            index
        }
    }
}

impl<T> IndexScopedCoreTrait<T> for IndexScopedCoreObj<'_, T> {
    fn get_some_number(&self) -> i32 {
        self.wrapped.get_some_number(self.index)
    }
    fn set_some_number(&mut self, some_number: i32) {
        self.wrapped.set_some_number(some_number, self.index);
    }

    fn get_some_item(&self) -> &Option<T> {
        self.wrapped.get_some_item()
    }
    fn set_some_item(&mut self, some_item: Option<T>) {
        self.wrapped.set_some_item(some_item);
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_index_scoped_core_obj() {
        let mut core_obj = CoreObj::new([0, 1, 2], Some(3));
        let mut index_scoped_core_obj = IndexScopedCoreObj::new(&mut core_obj, 1);
        assert_eq!(index_scoped_core_obj.get_some_number(), 1);
        index_scoped_core_obj.set_some_number(4);
        assert_eq!(index_scoped_core_obj.get_some_number(), 4);
        assert_eq!(index_scoped_core_obj.get_some_item(), &Some(3));
        index_scoped_core_obj.set_some_item(Some(5));
        assert_eq!(index_scoped_core_obj.get_some_item(), &Some(5));

        assert_eq!(core_obj.get_some_number(1), 4);
        assert_eq!(core_obj.some_numbers, [0, 4, 2]);
    }
}