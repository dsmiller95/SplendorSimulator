use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::CardTier;
use rust_splendor::game_model::game_config::TieredCard;
use crate::components::resource_cost::SplendorResourceCost;

#[pyclass]
pub struct SplendorCard {
    wrapped: TieredCard,
}

impl From<TieredCard> for SplendorCard {
    fn from(card: TieredCard) -> Self {
        SplendorCard {
            wrapped: card,
        }
    }
}

#[pymethods]
impl SplendorCard {
    #[getter]
    fn get_id(&self) -> u32 {
        self.wrapped.card.id
    }
    #[getter]
    fn get_tier(&self) -> u8 {
        match self.wrapped.tier {
            CardTier::CardTier1 => 1,
            CardTier::CardTier2 => 2,
            CardTier::CardTier3 => 3
        }
    }
    #[getter]
    fn get_costs(&self) -> SplendorResourceCost {
        self.wrapped.card.cost.into()
    }
    #[getter]
    fn get_returns(&self) -> SplendorResourceCost {
        self.wrapped.card.returns.into()
    }
    #[getter]
    fn get_points(&self) -> i8 {
        self.wrapped.card.points
    }
}