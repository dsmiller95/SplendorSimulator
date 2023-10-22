use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::ResourceAmountFlags;
use rust_splendor::game_model::game_components::Card;

#[pyclass]
pub struct SplendorResourceCost {
    wrapped: ResourceAmountFlags,
}

#[pymethods]
impl SplendorResourceCost {

}