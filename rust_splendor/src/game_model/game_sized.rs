use crate::constants::{CARD_COUNT_PER_TIER, RESOURCE_TYPE_COUNT, MAX_RESERVED_CARDS, RESOURCE_TOKEN_COUNT};
use crate::game_model::game_components::Card;

#[derive(Debug)]
pub struct CardRowSized {
    pub open_cards: [Option<Card>; CARD_COUNT_PER_TIER],
    /// the top of the pile. typically invisible to the players, but included here because
    /// it can be involved in executing a specific action.
    pub hidden_card: Option<Card>,
}


impl CardRowSized {
    pub fn new() -> CardRowSized {
        CardRowSized {
            open_cards: std::array::from_fn(|_| None),
            hidden_card: None,
        }
    }
}
