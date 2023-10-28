use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_components::{Noble};
use crate::components::resource_cost::SplendorResourceCost;

#[pyclass]
pub struct SplendorNoble {
    wrapped: Noble,
}

impl SplendorNoble {
    pub fn new(noble: Noble) -> SplendorNoble {
        SplendorNoble {
            wrapped: noble,
        }
    }
}

impl From<Noble> for SplendorNoble {
    fn from(noble: Noble) -> Self {
        SplendorNoble::new(noble)
    }
}

#[pymethods]
impl SplendorNoble {
    #[getter]
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.wrapped.id)
    }
    #[getter]
    fn get_costs(&self) -> PyResult<SplendorResourceCost> {
        Ok(self.wrapped.cost.into())
    }
    #[getter]
    fn get_points(&self) -> PyResult<i8> {
        Ok(self.wrapped.points)
    }
}