
pub trait ToVect {
    fn vect_size() -> usize{
        let mut result = 0;
        for desc in Self::describe_slice() {
            result += desc.size;
        }
        result
    }
    fn populate_slice(&self, slice: &mut [f32]);
    fn describe_slice() -> Vec<ToVectDescription>;
}

#[derive(Clone)]
pub struct ToVectDescription {
    pub name: Box<str>,
    pub size: usize,
}

pub fn get_n_descriptions<'a>(n: usize, name: &'a str, sub_descriptors: &'a Vec<ToVectDescription>) -> impl Iterator<Item=ToVectDescription> + 'a{
    (0..n).flat_map(move |i| {
        sub_descriptors.iter().map(move |sub_desc| {
            ToVectDescription{
                name: format!("{}_{}_{}", name, i, sub_desc.name).into(),
                size: sub_desc.size,
            }
        })
    })
}

use pyo3::prelude::*;
#[pyclass]
pub struct ToVectDescriptionIndexes{
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

pub fn rollup_indexes<'a>(descriptions: &'a Vec<ToVectDescription>) -> Vec<ToVectDescriptionIndexes>{
    let mut result = vec![];
    let mut offset = 0;
    for desc in descriptions.iter() {
        result.push(ToVectDescriptionIndexes{
            name: desc.name.to_string(),
            start: offset,
            end: offset+desc.size,
        });
        offset += desc.size;
    }
    result
}