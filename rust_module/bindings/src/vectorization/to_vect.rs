
pub trait ToVect {
    fn vect_size() -> usize;
    fn populate_slice(&self, slice: &mut [f32]);
}
