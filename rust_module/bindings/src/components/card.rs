use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_components::Card;

#[pyclass]
pub struct SplendorCard {
    wrapped: Card,
}

#[pymethods]
impl SplendorCard {
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.wrapped.id)
    }
}