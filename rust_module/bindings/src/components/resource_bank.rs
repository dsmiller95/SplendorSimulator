use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::{ResourceTokenBank, ResourceTokenType};
use crate::components::resource_type::{SplendorResourceType};

#[pyclass]
pub struct SplendorResourceBank {
    wrapped: ResourceTokenBank,
}

impl SplendorResourceBank {
    pub fn new() -> SplendorResourceBank {
        SplendorResourceBank {
            wrapped: [0; 6],
        }
    }
    pub fn with_cost(mut self, cost: ResourceTokenBank) -> Self {
        self.wrapped = cost;
        self
    }
}

impl From<ResourceTokenBank> for SplendorResourceBank {
    fn from(cost: ResourceTokenBank) -> Self {
        SplendorResourceBank::new().with_cost(cost)
    }
}

#[pymethods]
impl SplendorResourceBank {
    #[pyo3(name = "__getitem__")]
    fn get_resource_amount(&self, idx: SplendorResourceType) -> i8 {
        let res_type : ResourceTokenType = idx.into();
        self.wrapped[res_type]
    }
}