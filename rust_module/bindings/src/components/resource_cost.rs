use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::{ResourceAmountFlags, ResourceType};
use crate::components::resource_type::{SplendorResourceType};

#[pyclass]
pub struct SplendorResourceCost {
    wrapped: ResourceAmountFlags,
}

impl SplendorResourceCost {
    pub fn new() -> SplendorResourceCost {
        SplendorResourceCost {
            wrapped: [0; 5],
        }
    }
    pub fn with_cost(mut self, cost: ResourceAmountFlags) -> Self {
        self.wrapped = cost;
        self
    }
}

impl From<ResourceAmountFlags> for SplendorResourceCost {
    fn from(cost: ResourceAmountFlags) -> Self {
        SplendorResourceCost::new().with_cost(cost)
    }
}

#[pymethods]
impl SplendorResourceCost {
    #[pyo3(name = "__getitem__")]
    fn get_resource_amount(&self, idx: SplendorResourceType) -> PyResult<i8> {
        let res_type : ResourceType = SplendorResourceType::try_into(idx)?;
        Ok(self.wrapped[res_type])
    }
}