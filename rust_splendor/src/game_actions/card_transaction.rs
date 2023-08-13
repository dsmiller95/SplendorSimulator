use crate::constants::{CardPickOnBoard, GlobalCardPick, MAX_RESERVED_CARDS, PlayerSelection, reserved_card, ReservedCardSelection, ResourceTokenType::*};
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData, PutError};
use crate::game_model::game_components::Card;

/// check if the card can move from one place to another
/// does not check to see if the player has the resources to do so
pub fn can_transact<ActorType, T>(game: &T, transaction: &CardTransaction) -> Result<(), CardTransactionError>
    where ActorType: KnowableActorData,
          T: KnowableGameData<ActorType>
{
    game.get_card_pick(&transaction.get_card_pick())
        .ok_or(CardTransactionError::CardDoesNotExist)?;
    
    let player = game.get_actor_at_index(transaction.player)
        .ok_or(CardTransactionError::PlayerDoesNotExist)?;
    
    match transaction.selection_type {
        CardSelectionType::ObtainReserved(_) => {
            if player.iterate_reserved_cards().count() >= MAX_RESERVED_CARDS {
                return Err(CardTransactionError::MaximumReservedCardsExceeded);
            }
        }
        CardSelectionType::ObtainBoard(_) => {}
        CardSelectionType::Reserve(_) => {}
    }
    
    Ok(())
}

/// move the card from one place to another
/// does not check to see if the player has the resources to do so, nor does it modify the player's resources
pub fn transact<ActorType, T>(game: &mut T, transaction: &CardTransaction) -> Result<CardTransactionSuccess, CardTransactionError>
    where ActorType: KnowableActorData,
          T: KnowableGameData<ActorType>
{
    can_transact(game, transaction)?;

    let pick = transaction.get_card_pick();

    let card = game.take_card(&pick).ok_or(CardTransactionError::CardDoesNotExist)?;
    
    let actor = game.get_actor_at_index_mut(transaction.player)
        .ok_or(CardTransactionError::PlayerDoesNotExist)?;
    
    let consume_result = match transaction.selection_type {
        CardSelectionType::ObtainBoard(_) => {
            actor.put_in_purchased(card)
        }
        CardSelectionType::ObtainReserved(_) => {
            actor.put_in_purchased(card)
        }
        CardSelectionType::Reserve(_) => {
            actor.put_in_reserve(card)
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
            Err(CardTransactionError::UnkownCardOccupied)
        }
    }
    
    // match consume_result {
    //     Ok(_) => Ok(CardTransactionSuccess::FullTransaction),
    //     Err(e) => {
    //         let mapped = match e {
    //             TryConsumeError::SourceEmpty => CardTransactionError::CardDoesNotExist,
    //             TryConsumeError::DestinationOccupied => CardTransactionError::UnkownCardOccupied,
    //             TryConsumeError::DestinationDoesNotExist => CardTransactionError::PlayerDoesNotExist,
    //         };
    //         Err(mapped)
    //     }
    // }
}

impl CardTransaction{
    pub fn get_card_pick(&self) -> GlobalCardPick{
        match self.selection_type {
            CardSelectionType::Reserve(onBoard) => onBoard.into(),
            CardSelectionType::ObtainBoard(onBoard) => onBoard.into(),
            CardSelectionType::ObtainReserved(reserved) => reserved_card(self.player, reserved),
        }
    }
}

/// A transaction which will move a card from the board to the player's inventory, or to their reserved cards
pub struct CardTransaction{
    pub player: PlayerSelection,
    pub selection_type: CardSelectionType,
}

pub enum CardSelectionType{
    ObtainBoard(CardPickOnBoard),
    ObtainReserved(ReservedCardSelection),
    Reserve(CardPickOnBoard)
}

#[derive(Debug, PartialEq)]
pub enum CardTransactionError{
    MaximumReservedCardsExceeded,
    CardDoesNotExist,
    PlayerDoesNotExist,
    UnkownCardOccupied
}

#[derive(Debug, PartialEq)]
pub enum CardTransactionSuccess {
    FullTransaction,
    PartialTransaction
}


#[cfg(test)]
mod tests {
    use crate::constants::CardPickInTier::OpenCard;
    use crate::constants::CardTier::*;
    use crate::constants::OpenCardPickInTier::*;
    use crate::constants::PlayerSelection::*;
    use crate::game_actions::card_transaction::CardSelectionType::Reserve;
    use crate::game_actions::knowable_game_data::HasCards;
    use super::*;

    use super::CardTransactionError::*;
    use super::CardTransactionSuccess::*;
    
    
    #[test]
    fn singe_card_reserve_success() {
        let player_n = PlayerSelection3;
        let card_pick = CardPickOnBoard {
            tier: CardTier2,
            pick: OpenCard(OpenCardPickInTier1),
        };
        let card_id = 24;
        let card = Card::new().with_id(card_id);
        
        let mut game = crate::game_actions::test_utils::get_test_game(4).game_sized;
        game.try_put_card(&card_pick.into(), card).unwrap();
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 0);

        let transaction = CardTransaction{
            player: PlayerSelection1,
            selection_type: Reserve(card_pick),
        };

        let result = transact(&mut game, &transaction);

        assert_eq!(result, Ok(FullTransaction));

        let card = game.get_card_pick(&card_pick.into());
        assert!(card.is_none());
        
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);
        assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
        assert_eq!(actor.resource_tokens[Gold], 0);
    }
    
    #[test]
    fn singe_card_reserve_when_existing_reserve_success() {
        let player_n = PlayerSelection3;
        let card_pick = CardPickOnBoard {
            tier: CardTier2,
            pick: OpenCard(OpenCardPickInTier1),
        };
        let card_id = 24;
        let card = Card::new().with_id(card_id);

        let mut game = crate::game_actions::test_utils::get_test_game(4).game_sized;
        game.try_put_card(&card_pick.into(), card).unwrap();
        
        let actor = game.get_actor_at_index_mut(player_n).unwrap();
        actor.put_in_reserve(Card::new().with_id(1)).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);

        let transaction = CardTransaction{
            player: PlayerSelection1,
            selection_type: Reserve(card_pick),
        };

        let result = transact(&mut game, &transaction);

        assert_eq!(result, Ok(FullTransaction));

        let card = game.get_card_pick(&card_pick.into());
        assert!(card.is_none());

        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 2);
        assert_eq!(actor.iterate_reserved_cards().filter(|x| x.id == card_id).next().is_some(), true);
        assert_eq!(actor.resource_tokens[Gold], 0);
    }
}

