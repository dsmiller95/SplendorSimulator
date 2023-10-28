use crate::game_model::game_config::GameConfig;
use crate::game_model::game_full::GameModel;

pub fn get_test_game(player_count: usize) -> GameModel {
    let game_config = GameConfig::new();
    GameModel::new(game_config, player_count, None)
}