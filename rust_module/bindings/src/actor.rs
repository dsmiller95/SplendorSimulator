use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_actions::knowable_game_data::KnowableActorData;
use rust_splendor::game_model::actor::Actor;
use crate::components::resource_bank::SplendorResourceBank;

#[pyclass]
pub struct SplendorActor {
    wrapped: Actor,
}

impl From<Actor> for SplendorActor {
    fn from(card: Actor) -> Self {
        SplendorActor {
            wrapped: card,
        }
    }
}

#[pymethods]
impl SplendorActor {
    #[getter]
    fn get_points(&self) -> i8 {
        self.wrapped.get_points()
    }

    #[getter]
    fn get_resources(&self) -> SplendorResourceBank {
        self.wrapped.resource_tokens.into()
    }
}