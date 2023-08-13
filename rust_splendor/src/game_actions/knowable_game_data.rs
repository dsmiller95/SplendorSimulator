use crate::constants::{GlobalCardPick, MAX_INVENTORY_TOKENS, PlayerSelection, RESOURCE_TOKEN_COUNT, ResourceTokenType};
use crate::game_model::game_components::Card;

pub trait KnowableGameData<ActorType> : HasCards
    where ActorType: KnowableActorData
{
    fn get_actor_at_index(&self, index: PlayerSelection) -> Option<&ActorType>;
    fn get_actor_at_index_mut(&mut self, index: PlayerSelection) -> Option<&mut ActorType>;
    
    fn bank_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT];
    fn bank_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT];
}

pub trait KnowableActorData {
    fn owned_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT];
    fn owned_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT];
    
    fn can_afford_card(&self, card: &Card) -> bool;
    fn reserved_cards(&self) -> &[Option<Card>];
}


pub trait HasCards {
    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card>;
}
