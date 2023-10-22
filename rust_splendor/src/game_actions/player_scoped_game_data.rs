use crate::constants::{GlobalCardPick, MAX_RESERVED_CARDS, ResourceAmountFlags, ResourceTokenBank};
use crate::game_actions::knowable_game_data::PutError;
use crate::game_model::game_components::Card;

pub trait PlayerScopedGameData {

    fn bank_resources(&self) -> &ResourceTokenBank;

    fn owned_resources(&self) -> &ResourceTokenBank;

    fn persistent_resources(&self) -> &ResourceAmountFlags;
    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
    fn iterate_reserved_cards(&self) -> impl Iterator<Item = &Card>{
        self.reserved_cards()
            .iter()
            .filter_map(|card| card.as_ref())
    }


    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
}

pub trait PlayerScopedGameDataMut {
    fn bank_resources_mut(&mut self) -> &mut ResourceTokenBank;

    fn owned_resources_mut(&mut self) -> &mut ResourceTokenBank;

    fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
    fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;


    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card>;

    fn take_card(&mut self, card_pick: &GlobalCardPick) -> Option<Card>;
    fn try_put_card(&mut self, card_pick: &GlobalCardPick, card: Card) -> Result<(), PutError<Card>>;
}