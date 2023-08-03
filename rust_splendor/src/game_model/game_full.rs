use crate::game_model::constants::{CardPick};
use crate::game_model::game_components::Card;
use crate::game_model::game_config::GameConfig;
use crate::game_model::game_sized::GameSized;
use crate::game_model::game_unsized::GameUnsized;

pub struct GameModel {
    pub game_unsized: GameUnsized,
    pub game_sized: GameSized,
    pub game_config: GameConfig,
}

pub trait HasCards {
    fn get_card_pick(&self, card_pick: &CardPick) -> Option<&Card>;
    fn get_card_pick_mut(&mut self, card_pick: &CardPick) -> Option<&mut Card>;
}

impl HasCards for GameSized {
    fn get_card_pick(&self, card_pick: &CardPick) -> Option<&Card> {
        self.card_rows[card_pick.tier][card_pick.pick_in_tier].as_ref()
    }
    fn get_card_pick_mut(&mut self, card_pick: &CardPick) -> Option<&mut Card> {
        self.card_rows[card_pick.tier][card_pick.pick_in_tier].as_mut()
    }
}