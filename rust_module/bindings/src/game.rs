use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_full::GameModel;
use crate::config::SplendorConfig;

#[pyclass]
pub struct SplendorGame {
    wrapped_game: GameModel,
}

#[pymethods]
impl SplendorGame {
    // implicit clone on config
    #[new]
    fn new(config: SplendorConfig, player_n: usize) -> PyResult<SplendorGame> {
        Ok(SplendorGame {
            wrapped_game: GameModel::new(config.wrapped_config, player_n),
        })
    }


    fn get_packed_state_array(&self) -> PyResult<Vec<f32>> {
        Ok(vec![22.0])
    }
}