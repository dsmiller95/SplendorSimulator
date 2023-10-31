use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::CardTier::*;
use rust_splendor::constants::{CardPickInTier, CardPickOnBoard, GlobalCardPick, OpenCardPickInTier};
use rust_splendor::game_actions::knowable_game_data::HasCards;
use rust_splendor::game_model::game_config::TieredCard;
use rust_splendor::game_model::game_full::GameModel;
use crate::actor::SplendorActor;
use crate::components::card::SplendorCard;
use crate::components::resource_bank::SplendorResourceBank;
use crate::config::SplendorConfig;
use crate::turn::SplendorTurn;

#[pyclass]
pub struct SplendorGame {
    wrapped_game: GameModel,
}

#[pymethods]
impl SplendorGame {
    // implicit clone on config
    #[new]
    fn new(config: SplendorConfig, player_n: usize, rand_seed: Option<u64>) -> PyResult<SplendorGame> {
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

    fn get_card_row(&self, tier: u8) -> Vec<Option<SplendorCard>> {
        let tier = match tier {
            1 => CardTier1,
            2 => CardTier2,
            3 => CardTier3,
            _ => panic!("tier must be between 1 and 3 inclusive"),
        };
        OpenCardPickInTier::iterator()
            .map(|&x| {
                let global_pick: GlobalCardPick = CardPickOnBoard {
                    tier,
                    pick: CardPickInTier::OpenCard(x),
                }.into();

                self.wrapped_game
                    .get_card_pick(&global_pick)
                    .map(|x| TieredCard{
                        tier,
                        card: x.clone()
                    }.into()
                    )
            })
            .collect()
    }

    fn take_turn(&mut self, turn: &SplendorTurn) -> PyResult<()> {
        let turn = &turn.wrapped;

        todo!()
        // self.wrapped_game.take_turn(&turn).map_err(|e| {
        //     PyValueError::new_err(format!("Error taking turn: {}", e))
        //})
    }
}