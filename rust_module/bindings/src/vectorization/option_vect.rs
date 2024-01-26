use crate::vectorization::*;

impl<T: ToVect> ToVect for Option<T> {
    fn describe_slice() -> Vec<ToVectDescription>{
        todo!()
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        match self {
            Some(t) => t.populate_slice(slice),
            None => (),
        }
    }
}