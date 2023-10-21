use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};

use crate::game_actions::bank_transactions::{BankTransaction, transact};
use crate::game_actions::card_transaction::{CardTransaction, CardTransactionError, transact_card};
use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};

pub struct SubTurn{
    pub action: SubTurnAction,
    pub failure_mode: SubTurnFailureMode
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SubTurnFailureMode{
    MustAllSucceed,
    MayPartialSucceed
}

pub enum SubTurnAction{
    /// Perform all transactions with the global bank
    TransactTokens(Vec<BankTransaction>),
    /// Perform a card transaction
    TransactCard(CardTransaction)
}
impl SubTurnAction {
    pub fn do_sub_turn<T: KnowableGameData<ActorType>, ActorType : KnowableActorData>(&self, game: &mut T) -> Result<TurnSuccess, TurnFailed>{
        match self {
            SubTurnAction::TransactTokens(bank_transactions) => {
                let any_transact_failed = bank_transactions
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
            }
            SubTurnAction::TransactCard(reserve_transaction) => {
                transact_card(game, &reserve_transaction)
                    .map_err(|e| match e {
                        CardTransactionError::UnknownCardOccupied => TurnFailed::FailurePartialModification,
                        _ => TurnFailed::FailureNoModification
                    })
                    .map(|_| TurnSuccess::Success)
            }
        }
    }
}
