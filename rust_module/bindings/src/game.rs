use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::CardTier::*;
use rust_splendor::constants::{CardPickInTier, CardPickOnBoard, GlobalCardPick, OpenCardPickInTier, PlayerSelection};
use rust_splendor::game_actions::knowable_game_data::{HasCards, KnowableGameData};
use rust_splendor::game_actions::turn::GameTurn;
use rust_splendor::game_actions::turn_result::{TurnFailed};
use rust_splendor::game_model::game_config::TieredCard;
use rust_splendor::game_model::game_full::{GameCreationFailure, GameModel};
use crate::actor::SplendorActor;
use crate::components::card::SplendorCard;
use crate::components::resource_bank::SplendorResourceBank;
use crate::config::SplendorConfig;
use crate::turn::SplendorTurn;
use crate::vectorization::ToVect;

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
            wrapped_game: GameModel::new(config.wrapped_config, player_n, rand_seed)
                .map_err(|e| PyValueError::new_err(format!("cannot create game: {:?}", e)))?,
        })
    }
    // get an array of floats which represent the entire visible game state
    // intended to be used as input to a neural network
    fn get_packed_state_array(&self) -> PyResult<Vec<f32>> {
        let game_model = &self.wrapped_game;
        let vect_size = GameModel::vect_size();
        // construct a vector of floats set to 0.0
        let game_vector = vec![0.0; vect_size];
        Ok(game_vector)
    }

    #[getter]
    fn get_turn_n(&self) -> u32 {
        self.wrapped_game.total_turns_taken
    }
    #[getter]
    fn get_active_player_index(&self) -> usize {
        self.wrapped_game.active_player
    }
    #[getter]
    fn get_active_player(&self) -> SplendorActor {
        self.wrapped_game.get_active_player().clone().into()
    }
    fn get_player_at(&self, player_index: usize) -> PyResult<SplendorActor> {
        let player_index = PlayerSelection::iterator()
            .nth(player_index)
            .ok_or(PyValueError::new_err("player index out of bounds"))?
            .clone();

        let active_player =self.wrapped_game.get_actor_at_index(player_index)
            .ok_or(PyValueError::new_err("player index out of bounds"))?;

        Ok(active_player.clone().into())
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

    fn take_turn(&mut self, turn: &SplendorTurn) -> PyResult<String> {
        let turn = &turn.wrapped;

        let mut scoped = self.wrapped_game.scope_to_active_player()
            .ok_or(GameTurnFailure::MissingPlayer)?;
        let turn_result = turn.take_turn(&mut scoped)
            .map_err(|x| GameTurnFailure::TurnFailure(x))?;

        self.wrapped_game.advance_player();

        Ok(format!("Turn success: {:?}, next player: {:?}", turn_result, self.wrapped_game.active_player))
    }
}

#[derive(Debug)]
enum GameTurnFailure {
    MissingPlayer,
    TurnFailure(TurnFailed),
}

impl From<GameTurnFailure> for PyErr {
    fn from(value: GameTurnFailure) -> Self {
        use pyo3::exceptions::PyValueError;
        PyValueError::new_err(format!("Game turn error: {:?}", value))
    }
}