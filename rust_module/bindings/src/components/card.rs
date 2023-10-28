use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_config::TieredCard;
use crate::components::resource_cost::SplendorResourceCost;

#[pyclass]
pub struct SplendorCard {
    wrapped: TieredCard,
}

impl SplendorCard {
    pub fn new(card: TieredCard) -> SplendorCard {
        SplendorCard {
            wrapped: card,
        }
    }
}

impl From<TieredCard> for SplendorCard {
    fn from(card: TieredCard) -> Self {
        SplendorCard::new(card)
    }
}

#[pymethods]
impl SplendorCard {
    #[getter]
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.wrapped.card.id)
    }
    #[getter]
    fn get_tier(&self) -> PyResult<u8> {
        Ok(self.wrapped.tier)
    }
    #[getter]
    fn get_costs(&self) -> PyResult<SplendorResourceCost> {
        Ok(self.wrapped.card.cost.into())
    }
    #[getter]
    fn get_returns(&self) -> PyResult<SplendorResourceCost> {
        Ok(self.wrapped.card.returns.into())
    }
    #[getter]
    fn get_points(&self) -> PyResult<i8> {
        Ok(self.wrapped.card.points)
    }
}