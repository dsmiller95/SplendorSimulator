use crate::game_actions::bank_transactions::{BankTransaction};
use crate::game_actions::card_transaction::{CardTransaction, CardTransactionError, transact_card};
use crate::game_actions::player_scoped_game_data::PlayerScopedGameData;
use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};

#[derive(Debug, PartialEq)]
pub struct SubTurn{
    pub action: SubTurnAction,
    pub failure_mode: SubTurnFailureMode
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SubTurnFailureMode{
    MustAllSucceed,
    MayPartialSucceed
}

#[derive(Debug, PartialEq)]
pub enum SubTurnAction{
    /// Perform all transactions with the global bank
    TransactTokens(Vec<BankTransaction>),
    /// Perform a card transaction
    TransactCard(CardTransaction)
}
impl SubTurnAction {
    pub fn to_partial(self) -> SubTurn {
        SubTurn {
            action: self,
            failure_mode: SubTurnFailureMode::MayPartialSucceed
        }
    }
    pub fn to_required(self) -> SubTurn {
        SubTurn {
            action: self,
            failure_mode: SubTurnFailureMode::MustAllSucceed
        }
    }
    pub fn do_sub_turn<'a, T: PlayerScopedGameData<'a>>(&self, game: &mut T) -> Result<TurnSuccess, TurnFailed>{
        match self {
            SubTurnAction::TransactTokens(bank_transactions) => {
                let any_transact_failed = bank_transactions
                    .iter()
                    .map(|transaction|
                        transaction.transact(game).is_err()
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

impl SubTurn{
    pub fn can_complete<'a, T: PlayerScopedGameData<'a>>(&self, game: &T) -> bool {
        match &self.action {
            SubTurnAction::TransactTokens(bank_transactions) => match self.failure_mode {
                SubTurnFailureMode::MustAllSucceed => {
                    bank_transactions.iter().all(|transaction|
                        transaction.can_transact(game).is_ok()
                    )
                }
                SubTurnFailureMode::MayPartialSucceed => {
                    bank_transactions.iter().any(|transaction|
                        transaction.can_transact(game).is_ok()
                    )
                }
            }
            SubTurnAction::TransactCard(card_transaction) => {
                card_transaction.can_transact(game).is_ok()
            }
        }
    }
}
