use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::game_model::game_config::GameConfig;

#[pyclass]
#[derive(Clone)]
pub struct SplendorConfig {
    pub wrapped_config: GameConfig,
}

impl SplendorConfig {
    fn parse_config_csv(&self, data: String) -> PyResult<SplendorConfig> {
        let mut config = GameConfig::new();
        // Ok(SplendorConfig {
        //     wrapped_config: config,
        // });
        todo!("parse csv")
    }
}