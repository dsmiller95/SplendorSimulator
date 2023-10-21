use crate::game_model::game_config::GameConfig;
use crate::game_model::game_full::GameModel;
use crate::game_model::game_sized::GameSized;

pub fn get_test_game(player_count: usize) -> GameModel {
    let game_config = GameConfig::new();
    let game_sized = GameSized::new(player_count);


    GameModel::new(game_sized, game_config, player_count)
}