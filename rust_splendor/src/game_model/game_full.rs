use std::cmp::min;
use crate::constants::{CARD_TIER_COUNT, GlobalCardPick, MAX_NOBLES, MAX_PLAYER_COUNT, PlayerSelection, RESOURCE_TOKEN_COUNT};
use crate::game_actions::knowable_game_data::{HasCards, KnowableGameData, PutError};
use crate::game_model::actor::Actor;
use crate::game_model::card::CardRow;
use crate::game_model::game_components::{Card, Noble};
use crate::game_model::game_config::GameConfig;

#[derive(Debug)]
pub struct GameModel {
    pub game_config: GameConfig,


    pub total_turns_taken: u32,

    /// assume the 1st actor in this list is the player whose turn it is.
    pub actors: [Option<Actor>; MAX_PLAYER_COUNT],
    pub card_rows: [CardRow; CARD_TIER_COUNT],

    pub available_nobles: [Option<Noble>; MAX_NOBLES],
    pub bank_resources: [i8; RESOURCE_TOKEN_COUNT],
    pub card_rows_sized: [CardRow; CARD_TIER_COUNT],
}

impl GameModel {
    pub fn new(config: GameConfig, player_count: usize) -> GameModel {
        let min_count = min(player_count, MAX_PLAYER_COUNT);
        let actors = std::array::from_fn(|i| {
            if i < min_count {
                Some(Actor::new())
            } else {
                None
            }
        });

        GameModel {
            game_config: config,

            total_turns_taken: 0,
            card_rows: std::array::from_fn(|_| CardRow::new()),


            actors,
            available_nobles: std::array::from_fn(|_| Some(Noble::new())),
            bank_resources: [0; RESOURCE_TOKEN_COUNT],
            card_rows_sized: std::array::from_fn(|_| CardRow::new()),
        }
    }
}
impl GameModel {
    fn get_mut_card_slot(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Option<Card>> {
        let mut_ref = match card_pick {
            GlobalCardPick::OnBoard(card_pick) => {
                &mut self.card_rows_sized[card_pick.tier]
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

impl HasCards for GameModel {
    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card> {
        match card_pick {
            GlobalCardPick::OnBoard(card_pick) => {
                self.card_rows_sized[card_pick.tier]
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

impl KnowableGameData<Actor> for GameModel {
    fn get_actor_at_index(&self, index: PlayerSelection) -> Option<&Actor> {
        self.actors[index].as_ref()
    }

    fn get_actor_at_index_mut(&mut self, index: PlayerSelection) -> Option<&mut Actor> {
        self.actors[index].as_mut()
    }

    fn bank_resources(&self) -> &[i8; 6] {
        &self.bank_resources
    }

    fn bank_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT] {
        &mut self.bank_resources
    }
}