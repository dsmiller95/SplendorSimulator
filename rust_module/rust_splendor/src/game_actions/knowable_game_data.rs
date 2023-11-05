use crate::constants::{GlobalCardPick, MAX_RESERVED_CARDS, PlayerSelection, RESOURCE_TOKEN_COUNT, ResourceAmountFlags};
use crate::game_model::game_components::Card;



pub trait KnowableGameData<ActorType> : HasCards
    where ActorType: KnowableActorData
{
    fn get_actor_at_index(&self, index: PlayerSelection) -> Option<&ActorType>;
    fn get_actor_at_index_mut(&mut self, index: PlayerSelection) -> Option<&mut ActorType>;
    
    fn get_active_player_selection(&self) -> PlayerSelection;
    
    fn bank_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT];
    fn bank_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT];
}

pub trait KnowableActorData {
    fn owned_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT];
    fn owned_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT];

    fn persistent_resources(&self) -> &ResourceAmountFlags;

    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
    
    fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
    fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;
    
    fn iterate_reserved_cards(&self) -> impl Iterator<Item = &Card>{
        self.reserved_cards()
            .iter()
            .filter_map(|card| card.as_ref())
    }

    fn get_points(&self) -> i8;
}


pub trait HasCards {
    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card>;
    
    fn take_card(&mut self, card_pick: &GlobalCardPick) -> Option<Card>;
    fn try_put_card(&mut self, card_pick: &GlobalCardPick, card: Card) -> Result<(), PutError<Card>>;
}

#[derive(Debug, Copy, Clone)]
pub enum PutError<T> {
    DestinationDoesNotExist(T),
    Occupied(T)
}
