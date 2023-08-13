use crate::constants::{GlobalCardPick, MAX_RESERVED_CARDS, PlayerSelection, RESOURCE_TOKEN_COUNT};
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
    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS];
    
    fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>>;
    fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>>;
    
    fn iterate_reserved_cards(&self) -> impl Iterator<Item = &Card>{
        self.reserved_cards()
            .iter()
            .filter_map(|card| card.as_ref())
    }
}


pub trait HasCards {
    fn get_card_pick(&self, card_pick: &GlobalCardPick) -> Option<&Card>;
    fn get_card_pick_mut(&mut self, card_pick: &GlobalCardPick) -> Option<&mut Card>;
    
    fn take_card(&mut self, card_pick: &GlobalCardPick) -> Option<Card>;
    fn try_put_card(&mut self, card_pick: &GlobalCardPick, card: Card) -> Result<(), PutError<Card>>;
    
    // TODO: might be kinda useless. since typically we will transfer to an actor that is part of self,
    //  so we won't be able to use a function like this due to ownership/borrowing rules.
    fn try_consume_card<F>(&mut self, card_pick: &GlobalCardPick, consumer: F) -> Result<(), TryConsumeError>
        where F:  FnOnce(Card) -> Result<(), PutError<Card>> {
        let card = self.take_card(card_pick).ok_or(TryConsumeError::SourceEmpty)?;
        
        let consumed_result = consumer(card);

        match consumed_result {
            Ok(()) => Ok(()),
            Err(PutError::DestinationDoesNotExist(card)) =>{
                self.try_put_card(card_pick, card).expect("Card slot must be empty, we just took a card from it");
                Err(TryConsumeError::DestinationDoesNotExist)  
            } 
            Err(PutError::Occupied(card)) => {
                self.try_put_card(card_pick, card).expect("Card slot must be empty, we just took a card from it");
                Err(TryConsumeError::DestinationOccupied)
            }
        }
    }
}

pub enum TryConsumeError {
    SourceEmpty,
    DestinationOccupied,
    DestinationDoesNotExist,
}

#[derive(Debug, Copy, Clone)]
pub enum PutError<T> {
    DestinationDoesNotExist(T),
    Occupied(T)
}
