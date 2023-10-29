use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_full::GameModel;
use crate::actor::SplendorActor;
use crate::components::resource_bank::SplendorResourceBank;
use crate::config::SplendorConfig;

#[pyclass]
pub struct SplendorGame {
    wrapped_game: GameModel,
}

#[pymethods]
impl SplendorGame {
    // implicit clone on config
    #[new]
    fn new(config: SplendorConfig, player_n: usize, rand_seed: Option<i64>) -> PyResult<SplendorGame> {
        Ok(SplendorGame {
            wrapped_game: GameModel::new(config.wrapped_config, player_n, rand_seed),
        })
    }
    fn get_packed_state_array(&self) -> PyResult<Vec<f32>> {
        Ok(vec![22.0])
    }

    #[getter]
    fn get_turn_n(&self) -> u32 {
        self.wrapped_game.total_turns_taken
    }
    #[getter]
    fn get_active_player_index(&self) -> u32 {
        self.wrapped_game.total_turns_taken
    }
    #[getter]
    fn get_active_player(&self) -> SplendorActor {
        self.wrapped_game.get_active_player().clone().into()
    }
    #[getter]
    fn get_bank(&self) -> SplendorResourceBank {
        self.wrapped_game.bank_resources.into()
    }
}