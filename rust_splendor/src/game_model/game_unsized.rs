use crate::game_model::constants::{CARD_TIER_COUNT};
use crate::game_model::game_components::{Card, Noble};
use super::constants::MAX_PLAYER_COUNT;

#[derive(Debug)]
pub struct GameUnsized {
    pub total_turns_taken: u32,
    
    pub actors: [Option<ActorUnsized>; MAX_PLAYER_COUNT],
    pub card_rows: [CardRowUnsized; CARD_TIER_COUNT],
}

#[derive(Debug)]
pub struct ActorUnsized {
    pub purchased_cards: Vec<Card>,
    pub claimed_nobles: Vec<Noble>,
}

#[derive(Debug)]
pub struct CardRowUnsized {
    /// all cards in the draw pile for the row, element 0 is top of the pile.
    pub hidden_cards: Vec<Card>,
}
