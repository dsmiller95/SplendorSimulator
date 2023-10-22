use std::marker::PhantomData;
use crate::constants::{GlobalCardPick, MAX_RESERVED_CARDS, PlayerSelection, ResourceAmountFlags, ResourceTokenBank};
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData, PutError};
use crate::game_model::game_components::Card;

pub struct PlayerScopedGameDataWrapper<'a, T: KnowableGameData<ActorType>, ActorType: KnowableActorData>
{
    game: &'a mut T,
    player: PlayerSelection,
    phantom: PhantomData<&'a ActorType>
}

impl<'a, T: KnowableGameData<ActorType>, ActorType : KnowableActorData> PlayerScopedGameDataWrapper<'a, T, ActorType> {
    pub fn new(game: &'a mut T, player: PlayerSelection) -> Option<Self> {
        if game.get_actor_at_index(player).is_none() {
            return None;
        }
        Some(Self {
            game,
            player,
            phantom: PhantomData
        })
    }

    fn get_actor_mut(&mut self) -> &mut ActorType {
        self.game
            .get_actor_at_index_mut(self.player)
            .expect("Player is validated to exist on wrapper construction")
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
            fn bank_resources_mut(&mut self) -> &mut ResourceTokenBank;

            fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
            fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card>;

            fn take_card(&mut self, card_pick: &GlobalCardPick) -> Option<Card>;
            fn try_put_card(&mut self, card_pick: &GlobalCardPick, card: Card) -> Result<(), PutError<Card>>;
        }
        to self.get_actor() {
            fn owned_resources(&self) -> &ResourceTokenBank;

            fn persistent_resources(&self) -> &ResourceAmountFlags;
            fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
        }
        to self.get_actor_mut(){
            fn owned_resources_mut(&mut self) -> &mut ResourceTokenBank;
            fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
            fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;
        }
    }
}