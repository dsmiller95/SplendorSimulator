use crate::vectorization::ToVect;

impl<T: ToVect> ToVect for Option<T> {
    fn vect_size() -> usize { 
        T::vect_size()
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        match self {
            Some(t) => t.populate_slice(slice),
            None => (),
        }
    }
}