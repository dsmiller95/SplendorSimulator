use crate::constants::CardTier;
use crate::game_model::game_components::{Card, Noble};

#[derive(Debug, Clone)]
pub struct GameConfig {
    pub all_cards: Vec<TieredCard>,
    pub all_nobles: Vec<Noble>,
    
    pub max_resource_tokens: i8,
}

#[derive(Debug, Clone)]
pub struct TieredCard{
    pub tier: CardTier,
    pub card: Card,
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