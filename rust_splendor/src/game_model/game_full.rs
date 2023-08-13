use crate::constants::{GlobalCardPick, PlayerSelection, RESOURCE_TOKEN_COUNT};
use crate::game_actions::knowable_game_data::{HasCards, KnowableGameData};
use crate::game_model::game_components::Card;
use crate::game_model::game_config::GameConfig;
use crate::game_model::game_sized::{ActorSized, GameSized};
use crate::game_model::game_unsized::GameUnsized;

pub struct GameModel {
    pub game_unsized: GameUnsized,
    pub game_sized: GameSized,
    pub game_config: GameConfig,
}

impl HasCards for GameSized {
    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card> {
        match card_pick {
            GlobalCardPick::OnBoard(card_pick) => {
                self.card_rows[card_pick.tier]
                    [card_pick.pick]
                    .as_ref()
            }
            GlobalCardPick::Reserved(reserved) => {
                self.actors[reserved.player_index].as_ref()?
                    .reserved_cards[reserved.reserved_card_index]
                    .as_ref()
            }
        }
    }
    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card> {
        match card_pick {
            GlobalCardPick::OnBoard(card_pick) => {
                self.card_rows[card_pick.tier]
                    [card_pick.pick]
                    .as_mut()
            }
            GlobalCardPick::Reserved(reserved) => {
                self.actors[reserved.player_index].as_mut()?
                    .reserved_cards[reserved.reserved_card_index]
                    .as_mut()
            }
        }
    }
}

impl KnowableGameData<ActorSized> for GameSized {
    fn get_actor_at_index(&self, index: PlayerSelection) -> Option<&ActorSized> {
        self.actors[index].as_ref()
    }

    fn get_actor_at_index_mut(&mut self, index: PlayerSelection) -> Option<&mut ActorSized> {
        self.actors[index].as_mut()
    }

    fn bank_resources(&self) -> &[i8; 6] {
        &self.bank_resources
    }

    fn bank_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT] {
        &mut self.bank_resources
    }
}