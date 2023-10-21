use crate::game_model::game_components::{Card, Noble};



#[derive(Debug)]
pub struct CardRowUnsized {
    /// all cards in the draw pile for the row, element 0 is top of the pile.
    pub hidden_cards: Vec<Card>,
}

impl CardRowUnsized {
    pub fn new() -> CardRowUnsized {
        CardRowUnsized {
            hidden_cards: Vec::new(),
        }
    }
}
