use crate::game_model::game_config::GameConfig;
use crate::game_model::game_full::GameModel;
use crate::game_model::game_sized::GameSized;
use crate::game_model::game_unsized::GameUnsized;

pub fn get_test_game(player_count: usize) -> GameModel {
    let game_config = GameConfig::new();
    let game_unsized = GameUnsized::new(player_count);
    let mut game_sized = GameSized::new(player_count);
    
    
    GameModel {
        game_unsized,
        game_sized,
        game_config,
    }
}