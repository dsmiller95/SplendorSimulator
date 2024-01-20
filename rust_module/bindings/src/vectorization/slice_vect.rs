use crate::vectorization::*;

impl<T: ToVect, const SIZE: usize> ToVect for [T; SIZE] {
    fn vect_size() -> usize { 
        T::vect_size() * SIZE
    }
    fn describe_slice() -> Vec<ToVectDescription>{
        todo!()
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        for (i, t) in self.iter().enumerate() {
            t.populate_slice(&mut slice[i*T::vect_size()..(i+1)*T::vect_size()]);
        }
    }
}