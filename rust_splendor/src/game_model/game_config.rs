use crate::game_model::game_components::{Card, Noble};

#[derive(Debug)]
pub struct GameConfig {
    pub all_cards: Vec<Card>,
    pub all_nobles: Vec<Noble>,
    
    pub max_resource_tokens: i8,
}

impl GameConfig {
    pub fn new() -> GameConfig {
        GameConfig {
            all_cards: Vec::new(),
            all_nobles: Vec::new(),
            max_resource_tokens: 0,
        }
    }
}
