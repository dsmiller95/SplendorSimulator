use std::marker::PhantomData;
use crate::constants::{MAX_RESERVED_CARDS, PlayerSelection, ResourceAmountFlags, ResourceTokenBank};
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData, PutError};
use crate::game_model::game_components::Card;

pub struct PlayerScopedGameDataWrapper<'a, T: KnowableGameData<ActorType>, ActorType: KnowableActorData>
{
    game: &'a mut T,
    player: PlayerSelection,
    phantom: PhantomData<ActorType>
}

impl<'a, T: KnowableGameData<ActorType>, ActorType : KnowableActorData> PlayerScopedGameDataWrapper<'_, T, ActorType> {
    fn get_actor(&self) -> &ActorType {
        self.game
            .get_actor_at_index(self.player)
            .expect("Player is validated to exist on wrapper construction")
    }
    fn get_actor_mut(&mut self) -> &mut ActorType {
        self.game
            .get_actor_at_index_mut(self.player)
            .expect("Player is validated to exist on wrapper construction")
    }
}

use delegate::delegate;
use crate::game_actions::player_scoped_game_data::{CanPlayerScope, PlayerScopedGameData};
use crate::game_model::actor::Actor;
use crate::game_model::game_full::GameModel;
use crate::constants::PlayerCardPick;


impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> PlayerScopedGameData<'_> for PlayerScopedGameDataWrapper<'_, T, ActorType> {
    delegate! {
        to self.game {
            fn bank_resources(&self) -> &ResourceTokenBank;
            fn bank_resources_mut(&mut self) -> &mut ResourceTokenBank;

        }
        to self.get_actor() {
            fn owned_resources(&self) -> &ResourceTokenBank;

            fn persistent_resources(&self) -> &ResourceAmountFlags;
            fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
        }
        to self.get_actor_mut() {
            fn owned_resources_mut(&mut self) -> &mut ResourceTokenBank;

            fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
            fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;
        }
    }

    fn get_card_pick(&self, card_pick: &PlayerCardPick) -> Option<&Card>{
        self.game.get_card_pick(&card_pick.as_global(self.player))
    }
    fn get_card_pick_mut(&mut self, card_pick: &PlayerCardPick) -> Option<&mut Card>{
        self.game.get_card_pick_mut(&card_pick.as_global(self.player))
    }

    fn take_card(&mut self, card_pick: &PlayerCardPick) -> Option<Card>{
        self.game.take_card(&card_pick.as_global(self.player))
    }
    fn try_put_card(&mut self, card_pick: &PlayerCardPick, card: Card) -> Result<(), PutError<Card>>{
        self.game.try_put_card(&card_pick.as_global(self.player), card)
    }
}
impl CanPlayerScope for GameModel {
    type ScopedGameData<'a> = PlayerScopedGameDataWrapper<'a, GameModel, Actor>;

    fn scope_to(&mut self, player: PlayerSelection) -> Option<Self::ScopedGameData<'_>> {
        if self.get_actor_at_index(player).is_none() {
            return None;
        }
        Some(PlayerScopedGameDataWrapper::<GameModel, Actor>{
            game: self,
            player,
            phantom: PhantomData
        })
    }
}

impl GameModel {
    pub fn scope_to_active_player(&mut self) -> Option<<GameModel as CanPlayerScope>::ScopedGameData<'_>> {
        self.scope_to(self.get_active_player_selection())
    }
}