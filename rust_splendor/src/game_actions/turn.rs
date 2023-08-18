use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::constants::{CardPickOnBoard, GlobalCardPick, PlayerSelection, ResourceType, ResourceTokenType};
use crate::game_actions::bank_transactions::{BankTransaction, can_transact, transact};
use crate::game_actions::card_transaction::{CardSelectionType, CardTransaction, CardTransactionError, transact_card};

#[derive(Debug)]
pub enum Turn {
    TakeThreeTokens(ResourceType, ResourceType, ResourceType),
    TakeTwoTokens(ResourceType),
    PurchaseCard(GlobalCardPick),
    ReserveCard(CardPickOnBoard),
    Noop, // reserved for testing, player passes their turn
}

pub trait GameTurn<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> {
    fn take_turn(&self, game: &mut T, actor_index: PlayerSelection) -> Result<TurnSuccess, TurnFailed>;
    /// Perform any validation that can be applied to self-data alone
    fn is_valid(&self) -> bool;
    fn can_take_turn(&self, game: &T, actor_index: PlayerSelection) -> bool;
}

#[derive(Debug, PartialEq)]
pub enum TurnSuccess {
    /// The full and complete effects of the turn have been applied
    Success,
    /// A partial subset of the turn's effects have been applied, but the game state is still valid
    /// the partial application of effects composes a valid turn
    SuccessPartial,
}

#[derive(Debug, PartialEq)]
pub enum TurnFailed{
    /// The turn was not applied, and cannot be applied to this player and game state
    FailureNoModification,
    /// The turn was partially applied, and the modified game state is now invalid
    FailurePartialModification,
}


impl Turn {
    fn all_bank_transactions(&self, actor_index: PlayerSelection) -> Vec<BankTransaction> {
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*b), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*c), amount: -1},
                ]
            },
            Turn::TakeTwoTokens(a) => {
                vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                ]
            },
            Turn::PurchaseCard(_) => {
                vec![]
            },
            Turn::ReserveCard(_) => {
                vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::Gold, amount: -1},
                ]
            },
            Turn::Noop => {
                vec![]
            }
        }
    }
        
}
    
impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> GameTurn<T, ActorType> for Turn {
    fn take_turn(&self, game: &mut T, actor_index: PlayerSelection) -> Result<TurnSuccess, TurnFailed> {
        if !self.can_take_turn(game, actor_index) {
            return Err(TurnFailed::FailureNoModification)
        }
        
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let any_transact_failed = self.all_bank_transactions(actor_index)
                    .iter()
                    .map(|transaction| 
                        transact(game, transaction).is_err()
                    )
                    .reduce(|a, b| a || b);
                
                match any_transact_failed {
                    None => Ok(TurnSuccess::Success),
                    Some(true) => Ok(TurnSuccess::SuccessPartial),
                    Some(false) => Ok(TurnSuccess::Success)
                }
            },

            Turn::TakeTwoTokens(_) => {
                let any_transact_failed = self.all_bank_transactions(actor_index)
                    .iter().map(|transaction|
                    transact(game, transaction).is_err()
                )
                    .reduce(|a, b| a || b);

                match any_transact_failed {
                    None => Ok(TurnSuccess::Success), // no transactions attempted
                    Some(true) => Ok(TurnSuccess::SuccessPartial), // at least one err
                    Some(false) => Ok(TurnSuccess::Success) // no errors, all transactions succeeded
                }
            }
            Turn::PurchaseCard(_) => {
                todo!()
            }
            Turn::ReserveCard(reserved_card) => {
                let reserve_transaction = CardTransaction{
                    player: actor_index,
                    selection_type: CardSelectionType::Reserve(*reserved_card),
                };
                transact_card(game, &reserve_transaction).map_err(|e| match e {
                    CardTransactionError::UnknownCardOccupied => TurnFailed::FailurePartialModification,
                    _ => TurnFailed::FailureNoModification
                }).map(|_| TurnSuccess::Success)
            }
            Turn::Noop => Ok(TurnSuccess::Success)
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                a != b && a != c && b != c
            },
            _ => true,
        }
    }
    
    fn can_take_turn(&self, game: &T, actor_index: PlayerSelection) -> bool {
        if !GameTurn::<T, ActorType>::is_valid(self) {
            return false
        }
        let actor = game.get_actor_at_index(actor_index);
        if actor.is_none() {
            return false
        }
        let actor = actor.unwrap();
        
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                self.all_bank_transactions(actor_index)
                    .iter().any(|transaction| 
                        can_transact(game, transaction).is_ok()
                    )
            },
            Turn::TakeTwoTokens(a) => {
                self.all_bank_transactions(actor_index)
                    .iter().any(|transaction|
                    can_transact(game, transaction).is_ok()
                )
            },
            Turn::PurchaseCard(card) => {
                let picked_card = game.get_card_pick(card);
                if picked_card.is_none() {
                    return false
                }
                let picked_card = picked_card.unwrap();
                actor.can_afford_card(picked_card)
            },
            Turn::ReserveCard(card) => {
                let picked_card = game.get_card_pick(&GlobalCardPick::OnBoard(*card));
                
                picked_card.is_some()
                    && actor.reserved_cards().iter().any(|x| x.is_none())
            },
            Turn::Noop => true,
        }
    }
}