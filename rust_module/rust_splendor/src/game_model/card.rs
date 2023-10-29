use crate::constants::CARD_COUNT_PER_TIER;
use crate::game_model::game_components::Card;

#[derive(Debug)]
pub struct CardRow {
    /// The top card in the draw pile
    pub hidden_card: Option<Card>,
    /// all cards in the draw pile for the row, the last element is the 2nd element from the top of the draw pile.
    pub hidden_cards: Vec<Card>,

    pub open_cards: [Option<Card>; CARD_COUNT_PER_TIER],
}
impl CardRow {
    pub fn new() -> CardRow {
        CardRow {
            hidden_card: None,
            hidden_cards: Vec::new(),
            open_cards: std::array::from_fn(|_| None)
        }
    }

    pub fn fill_with_single(&mut self, card: Card) {
        for card_slot in self.open_cards.iter_mut() {
            if let None = card_slot {
                *card_slot = Some(card);
                return;
            }
        }
        if let None = self.hidden_card {
            self.hidden_card = Some(card);
            return;
        }

        self.hidden_cards.push(card);
    }
}