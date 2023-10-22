use std::marker::PhantomData;
use crate::constants::{GlobalCardPick, MAX_RESERVED_CARDS, PlayerSelection, ResourceAmountFlags, ResourceTokenBank};
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::game_model::game_components::Card;

pub struct PlayerScopedGameDataWrapper<'a, T: KnowableGameData<ActorType>, ActorType: KnowableActorData>
{
    game: &'a T,
    player: PlayerSelection,
    phantom: PhantomData<ActorType>
}

impl<'a, T: KnowableGameData<ActorType>, ActorType : KnowableActorData> PlayerScopedGameDataWrapper<'a, T, ActorType> {
    pub fn new(game: &'a T, player: PlayerSelection) -> Option<Self> {
        if game.get_actor_at_index(player).is_none() {
            return None;
        }
        Some(Self {
            game,
            player,
            phantom: PhantomData
        })
    }

    fn get_actor(&self) -> &ActorType {
        self.game
            .get_actor_at_index(self.player)
            .expect("Player is validated to exist on wrapper construction")
    }
}

use delegate::delegate;
use crate::game_actions::player_scoped_game_data::PlayerScopedGameData;

impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> PlayerScopedGameData for PlayerScopedGameDataWrapper<'_, T, ActorType> {
    delegate! {
        to self.game {
            fn bank_resources(&self) -> &ResourceTokenBank;

            fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
        }
        to self.get_actor() {
            fn owned_resources(&self) -> &ResourceTokenBank;

            fn persistent_resources(&self) -> &ResourceAmountFlags;
            fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
        }
    }
}