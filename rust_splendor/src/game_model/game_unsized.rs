use crate::game_model::game_components::{Card, Noble};

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
