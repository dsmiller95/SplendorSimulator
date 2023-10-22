use crate::constants::{MAX_RESERVED_CARDS, PlayerCardPick, PlayerSelection, ResourceAmountFlags, ResourceTokenBank};
use crate::game_actions::knowable_game_data::PutError;
use crate::game_model::game_components::Card;

pub trait PlayerScopedGameData {

    fn bank_resources(&self) -> &ResourceTokenBank;
    fn bank_resources_mut(&mut self) -> &mut ResourceTokenBank;

    fn owned_resources(&self) -> &ResourceTokenBank;
    fn owned_resources_mut(&mut self) -> &mut ResourceTokenBank;

    fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
    fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;

    fn persistent_resources(&self) -> &ResourceAmountFlags;
    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
    fn iterate_reserved_cards(&self) -> impl Iterator<Item = &Card>{
        self.reserved_cards()
            .iter()
            .filter_map(|card| card.as_ref())
    }


    fn get_card_pick(&self, card_pick: &PlayerCardPick) -> Option<&Card>;
    fn get_card_pick_mut(&mut self, card_pick: &PlayerCardPick) -> Option<&mut Card>;
    fn take_card(&mut self, card_pick: &PlayerCardPick) -> Option<Card>;
    fn try_put_card(&mut self, card_pick: &PlayerCardPick, card: Card) -> Result<(), PutError<Card>>;
}

pub trait CanPlayerScope: Sized {
    type ScopedGameData: PlayerScopedGameData;
    fn scope_to(self, player: PlayerSelection) -> Option<Self::ScopedGameData>;
    fn extract_self(scoped: Self::ScopedGameData) -> Self;

    fn on_player<T>(self, player:PlayerSelection, f: impl FnOnce(&mut Self::ScopedGameData) -> T) -> (Self, T) {
        let mut scoped = self.scope_to(player)
            .expect("Player must exist");
        let result = f(&mut scoped);
        (Self::extract_self(scoped), result)
    }
}
