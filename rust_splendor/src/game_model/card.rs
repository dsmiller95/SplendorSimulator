use crate::constants::CARD_COUNT_PER_TIER;
use crate::game_model::game_components::Card;

#[derive(Debug)]
pub struct CardRow {
    /// all cards in the draw pile for the row, element 0 is top of the pile.
    pub hidden_cards: Vec<Card>,

    pub open_cards: [Option<Card>; CARD_COUNT_PER_TIER],

    /// the top of the pile. typically invisible to the players, but included here because
    /// it can be involved in executing a specific action.
    pub hidden_card: Option<Card>,
}
impl CardRow {
    pub fn new() -> CardRow {
        CardRow {
            hidden_cards: Vec::new(),
            open_cards: std::array::from_fn(|_| None),
            hidden_card: None,
        }
    }
}