use crate::constants::{MAX_RESERVED_CARDS, RESOURCE_TOKEN_COUNT, ResourceAmountFlags};
use crate::game_actions::knowable_game_data::{KnowableActorData, PutError};
use crate::game_model::actor::Actor;
use crate::game_model::game_components::Card;

impl KnowableActorData for Actor {
    fn owned_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT] {
        &self.resource_tokens
    }

    fn owned_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT] {
        &mut self.resource_tokens
    }

    fn persistent_resources(&self) -> &ResourceAmountFlags {
        &self.resources_from_cards
    }

    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS] {
        &self.reserved_cards
    }

    fn put_in_reserve(&mut self, card: Card) -> Result<(), PutError<Card>> {
        let first = self.reserved_cards.iter_mut()
            .filter(|x| x.is_none()).next();
        match first {
            Some(x) => {
                *x = Some(card);
                Ok(())
            }
            None => Err(PutError::DestinationDoesNotExist(card))
        }
    }

    fn put_in_purchased(&mut self, card: Card) -> Result<(), PutError<Card>> {
        self.purchased_cards.push(card);
        Ok(())
    }

    fn get_points(&self) -> i8 {
        // sum of points from cards and nobles
        self.purchased_cards.iter().map(|x| x.points).sum::<i8>() +
            self.claimed_nobles.iter().map(|x| x.points).sum::<i8>()
    }
}
#[cfg(test)]
mod tests {
    use crate::game_actions::knowable_game_data::KnowableActorData;
    use crate::game_model::actor::Actor;
    use crate::game_model::game_components::Card;

    #[test]
    fn puts_card_in_empty_reserve(){
        let mut actor = Actor::new();
        let card = Card::new().with_id(2);
        actor.put_in_reserve(card).unwrap();
        assert_eq!(actor.reserved_cards[0].as_ref().unwrap().id, 2);
    }
    #[test]
    fn puts_card_in_partiall_full_reserve(){
        let mut actor = Actor::new();
        actor.reserved_cards[0] = Some(Card::new().with_id(1));
        let card = Card::new().with_id(2);
        actor.put_in_reserve(card).unwrap();
        assert_eq!(actor.reserved_cards[1].as_ref().unwrap().id, 2);
    }
    
    #[test]
    fn cannot_puts_card_in_full_reserve(){
        let mut actor = Actor::new();
        actor.reserved_cards = std::array::from_fn(|i| Some(Card::new().with_id((i + 22) as u32)));
        let card = Card::new().with_id(33);
        actor.put_in_reserve(card).expect_err("Should not be able to put card in full reserve");
    }
    
}