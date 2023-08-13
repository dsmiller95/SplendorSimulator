use std::cmp::min;
use crate::constants::{CARD_TIER_COUNT};
use crate::game_model::game_components::{Card, Noble};
use crate::constants::MAX_PLAYER_COUNT;

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


impl GameUnsized {
    pub fn new(player_count: usize) -> GameUnsized {
        let min_count = min(player_count, MAX_PLAYER_COUNT);
        let actors = std::array::from_fn(|i| {
            if i < min_count {
                Some(ActorUnsized::new())
            } else {
                None
            }
        });
        GameUnsized {
            total_turns_taken: 0,
            actors,
            card_rows: std::array::from_fn(|_| CardRowUnsized::new()),
        }
    }
}

impl ActorUnsized {
    pub fn new() -> ActorUnsized {
        ActorUnsized {
            purchased_cards: Vec::new(),
            claimed_nobles: Vec::new(),
        }
    }
}


impl CardRowUnsized {
    pub fn new() -> CardRowUnsized {
        CardRowUnsized {
            hidden_cards: Vec::new(),
        }
    }
}
