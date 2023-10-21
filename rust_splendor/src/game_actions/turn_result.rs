use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::constants::{CardPickOnBoard, GlobalCardPick, PlayerSelection, ResourceType, ResourceTokenType};
use crate::game_actions::bank_transactions::{BankTransaction, can_transact, transact};
use crate::game_actions::card_transaction::{CardSelectionType, CardTransaction, CardTransactionError, transact_card};
#[derive(Debug, PartialEq)]
pub enum TurnSuccess {
    /// The full and complete effects of the turn have been applied
    Success,
    /// A partial subset of the turn's effects have been applied, but the game state is still valid
    /// the partial application of effects composes a valid turn
    SuccessPartial,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TurnFailed{
    /// The turn was not applied, and cannot be applied to this player and game state
    FailureNoModification,
    /// The turn was partially applied, and the modified game state is now invalid
    FailurePartialModification,
}

impl TurnSuccess {
    pub(crate) fn combine(&self, other: &TurnSuccess) -> TurnSuccess {
        match (self, other) {
            (TurnSuccess::Success, TurnSuccess::Success) => TurnSuccess::Success,
            _ => TurnSuccess::SuccessPartial
        }
    }
}
