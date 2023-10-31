use crate::constants::{CardPickOnBoard, MAX_RESERVED_CARDS, PlayerCardPick, ReservedCardSelection};
use crate::game_actions::knowable_game_data::{PutError};
use crate::game_actions::player_scoped_game_data::PlayerScopedGameData;

/// check if the card can move from one place to another
/// does not check to see if the player has the resources to do so
pub fn can_transact_card<'a, T: PlayerScopedGameData<'a>>(game: &T, transaction: &CardTransaction) -> Result<(), CardTransactionError>
{
    game.get_card_pick(&transaction.get_card_pick())
        .ok_or(CardTransactionError::CardDoesNotExist)?;

    match transaction.selection_type {
        CardSelectionType::ObtainReserved(_) => {
            if game.iterate_reserved_cards().count() >= MAX_RESERVED_CARDS {
                return Err(CardTransactionError::MaximumReservedCardsExceeded);
            }
        }
        CardSelectionType::ObtainBoard(_) => {}
        CardSelectionType::Reserve(_) => {}
    }
    
    Ok(())
}

impl CardTransaction {
    pub fn can_transact<'a, T: PlayerScopedGameData<'a>>(&self, game: &T) -> Result<(), CardTransactionError>{
        can_transact_card(game, self)
    }
}

/// move the card from one place to another
/// does not check to see if the player has the resources to do so, nor does it modify the player's resources
pub fn transact_card<'a, T: PlayerScopedGameData<'a>>(game: &mut T, transaction: &CardTransaction) -> Result<CardTransactionSuccess, CardTransactionError>
{
    can_transact_card(game, transaction)?;

    let pick = transaction.get_card_pick();

    let card = game.take_card(&pick).ok_or(CardTransactionError::CardDoesNotExist)?;

    let consume_result = match transaction.selection_type {
        CardSelectionType::ObtainBoard(_) => {
            game.put_in_purchased(card)
        }
        CardSelectionType::ObtainReserved(_) => {
            game.put_in_purchased(card)
        }
        CardSelectionType::Reserve(_) => {
            game.put_in_reserve(card)
        }
    };

    match consume_result {
        Ok(()) => Ok(CardTransactionSuccess::FullTransaction),
        Err(PutError::DestinationDoesNotExist(card)) =>{
            game.try_put_card(&pick, card).expect("Card slot must be empty, we just took a card from it");
            Err(CardTransactionError::PlayerDoesNotExist)
        }
        Err(PutError::Occupied(card)) => {
            game.try_put_card(&pick, card).expect("Card slot must be empty, we just took a card from it");
            Err(CardTransactionError::UnknownCardOccupied)
        }
    }
}

impl CardTransaction {
    pub fn get_card_pick(&self) -> PlayerCardPick{
        match self.selection_type {
            CardSelectionType::Reserve(on_board) => on_board.into(),
            CardSelectionType::ObtainBoard(on_board) => on_board.into(),
            CardSelectionType::ObtainReserved(reserved) => PlayerCardPick::Reserved(reserved),
        }
    }
}

/// A transaction which will move a card from the board
/// to the player's inventory or to their reserved cards
#[derive(Debug, PartialEq)]
pub struct CardTransaction {
    pub selection_type: CardSelectionType,
}

#[derive(Debug, PartialEq)]
pub enum CardSelectionType {
    ObtainBoard(CardPickOnBoard),
    ObtainReserved(ReservedCardSelection),
    Reserve(CardPickOnBoard)
}

#[derive(Debug, PartialEq)]
pub enum CardTransactionError{
    MaximumReservedCardsExceeded,
    CardDoesNotExist,
    PlayerDoesNotExist,
    UnknownCardOccupied
}

#[derive(Debug, PartialEq)]
pub enum CardTransactionSuccess {
    FullTransaction
}


#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::constants::CardPickInTier::OpenCard;
    use crate::constants::CardTier::*;
    use crate::constants::OpenCardPickInTier::*;
    use crate::constants::PlayerSelection::*;
    use crate::constants::ResourceTokenType::*;
    use crate::game_actions::card_transaction::CardSelectionType::Reserve;
    use crate::game_actions::knowable_game_data::HasCards;
    use crate::game_actions::player_scoped_game_data::CanPlayerScope;
    use super::*;
    use crate::game_model::game_components::Card;
    use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};


    use super::CardTransactionSuccess::*;
    
    
    #[test]
    fn when_empty_reserve__reserves_card() {
        let player_n = PlayerSelection3;
        let card_pick = CardPickOnBoard {
            tier: CardTier2,
            pick: OpenCard(OpenCardPickInTier1),
        };
        let card_id = 24;
        let card = Card::new().with_id(card_id);
        
        let mut game = crate::game_actions::test_utils::get_test_game(4);
        game.try_put_card(&card_pick.into(), card).unwrap();
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 0);

        let transaction = CardTransaction{
            selection_type: Reserve(card_pick),
        };

        let (game, result) = game.on_player(player_n, |scoped| {
            transact_card(scoped, &transaction)
        });

        assert_eq!(result, Ok(FullTransaction));

        let card = game.get_card_pick(&card_pick.into());
        assert!(card.is_none());
        
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);
        assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
        assert_eq!(actor.resource_tokens[Gold], 0);
    }
    
    #[test]
    fn when_partial_reserve__reserves_card() {
        let player_n = PlayerSelection3;
        let card_pick = CardPickOnBoard {
            tier: CardTier2,
            pick: OpenCard(OpenCardPickInTier1),
        };
        let card_id = 24;
        let card = Card::new().with_id(card_id);

        let mut game = crate::game_actions::test_utils::get_test_game(4);
        game.try_put_card(&card_pick.into(), card).unwrap();
        
        let actor = game.get_actor_at_index_mut(player_n).unwrap();
        actor.put_in_reserve(Card::new().with_id(1)).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);

        let transaction = CardTransaction{
            selection_type: Reserve(card_pick),
        };

        let (game, result) = game.on_player(player_n, |scoped| {
            transact_card(scoped, &transaction)
        });

        assert_eq!(result, Ok(FullTransaction));

        let card = game.get_card_pick(&card_pick.into());
        assert!(card.is_none());

        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 2);
        assert_eq!(actor.iterate_reserved_cards().filter(|x| x.id == card_id).next().is_some(), true);
        assert_eq!(actor.resource_tokens[Gold], 0);
    }
}

