use std::cmp::min;
use crate::constants::{CARD_TIER_COUNT, GlobalCardPick, MAX_PLAYER_COUNT, PlayerSelection, RESOURCE_TOKEN_COUNT};
use crate::game_actions::knowable_game_data::{HasCards, KnowableGameData, PutError};
use crate::game_model::game_components::Card;
use crate::game_model::game_config::GameConfig;
use crate::game_model::game_sized::{ActorSized, GameSized};
use crate::game_model::game_unsized::{ActorUnsized, CardRowUnsized};

#[derive(Debug)]
pub struct GameModel {
    pub game_sized: GameSized,
    pub game_config: GameConfig,


    pub total_turns_taken: u32,

    pub actors: [Option<ActorUnsized>; MAX_PLAYER_COUNT],
    pub card_rows: [CardRowUnsized; CARD_TIER_COUNT],
}

impl GameModel {
    pub fn new(sized: GameSized, config: GameConfig, player_count: usize) -> GameModel {
        let min_count = min(player_count, MAX_PLAYER_COUNT);
        let actors = std::array::from_fn(|i| {
            if i < min_count {
                Some(ActorUnsized::new())
            } else {
                None
            }
        });


        GameModel {
            game_sized: sized,
            game_config: config,

            total_turns_taken: 0,
            actors,
            card_rows: std::array::from_fn(|_| CardRowUnsized::new()),
        }
    }
}
impl GameSized {
    fn get_mut_card_slot(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Option<Card>> {
        let mut_ref = match card_pick {
            GlobalCardPick::OnBoard(card_pick) => {
                &mut self.card_rows[card_pick.tier]
                    [card_pick.pick]
            }
            GlobalCardPick::Reserved(reserved) => {
                &mut self.actors[reserved.player_index]
                    .as_mut()?
                    .reserved_cards[reserved.reserved_card]
            }
        };
        Some(mut_ref)
    }
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
                    .reserved_cards[reserved.reserved_card]
                    .as_ref()
            }
        }
    }
    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card> {
        let mutable_slot = self.get_mut_card_slot(card_pick);
        mutable_slot?.as_mut()
    }

    fn take_card(&mut self, card_pick: &GlobalCardPick) -> Option<Card> {
        let mutable_slot = self.get_mut_card_slot(card_pick);
        mutable_slot?.take()
    }

    fn try_put_card(&mut self, card_pick: &GlobalCardPick, card: Card) -> Result<(), PutError<Card>> {
        let mutable_slot = self.get_mut_card_slot(card_pick);
        match mutable_slot {
            None => {
                Err(PutError::DestinationDoesNotExist(card))
            }
            Some(Some(_)) => {
                Err(PutError::Occupied(card))
            }
            Some(None) => {
                *mutable_slot.unwrap() = Some(card);
                Ok(())
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