use std::ops::Range;

pub trait MapsToTensorAbilities {
    //fn get_tensor_values(&self) -> Vec<f32>;
    fn build_mapping_description<'a>(builder: TensorDescriptionBuilder<'a>) -> TensorDescriptionBuilder<'a>;
}

pub struct TensorMapping {
    pub range: Range<usize>,
    pub name: String
}

pub struct TensorDescription {
    pub tensor_mapping: Vec<TensorMapping>,
}

pub struct TensorDescriptionBuilder<'a> {
    prefix: String,
    tensor_description: &'a mut TensorDescription,
}

impl<'a> TensorDescriptionBuilder<'a> {
    pub fn new(description: &mut TensorDescription) -> TensorDescriptionBuilder {
        TensorDescriptionBuilder {
            prefix: String::new(),
            tensor_description: description,
        }
    }

    pub fn with_scope<'b>(&'b mut self, prefix: &str) -> TensorDescriptionBuilder<'b>
    where 'a: 'b {
        TensorDescriptionBuilder {
            prefix: format!("{0}{1}_", self.prefix, prefix),
            tensor_description: self.tensor_description,
        }
    }

    pub fn append_mapping(&mut self, name: &str, size: usize) {
        let last_end = self.tensor_description.tensor_mapping
            .last()
            .map(|m| m.range.end)
            .unwrap_or(0);
        let range = last_end..(last_end + size);
        self.tensor_description.tensor_mapping.push(TensorMapping {
            range,
            name: format!("{0}{1}", self.prefix, name),
        });
    }
}

