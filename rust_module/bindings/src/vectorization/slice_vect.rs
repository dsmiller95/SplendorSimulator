use crate::vectorization::*;

impl<T: ToVect, const SIZE: usize> ToVect for [T; SIZE] {
    fn describe_slice() -> Vec<ToVectDescription>{
        todo!()
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        let stepSize = slice.len() / SIZE;
        for (i, t) in self.iter().enumerate() {
            t.populate_slice(&mut slice[i*stepSize..(i+1)*stepSize]);
        }
    }
}