use crate::game_model::game_components::{Card, Noble};

#[derive(Debug)]
pub struct GameConfig {
    pub all_cards: Vec<Card>,
    pub all_nobles: Vec<Noble>,
    
    pub max_resource_tokens: u8,
}
