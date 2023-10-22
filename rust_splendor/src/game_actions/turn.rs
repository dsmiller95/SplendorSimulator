use std::cmp::{max, min};
use crate::constants::{CardPickOnBoard, ResourceType, ResourceTokenType, PlayerCardPick};
use crate::constants::ResourceTokenType::CostType;
use crate::game_actions::bank_transactions::{BankTransaction, get_transaction_sequence};
use crate::game_actions::card_transaction::{CardSelectionType, CardTransaction};
use crate::game_actions::player_scoped_game_data::PlayerScopedGameData;
use crate::game_actions::sub_turn::{SubTurn, SubTurnAction};
use crate::game_actions::sub_turn::SubTurnFailureMode::MayPartialSucceed;
use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};
use crate::game_actions::turn_result::TurnFailed::FailurePartialModification;

// TODO: how should we represent discarding tokens as part of a turn?
#[derive(Debug)]
pub enum Turn {
    TakeThreeTokens(ResourceType, ResourceType, ResourceType),
    TakeTwoTokens(ResourceType),
    PurchaseCard(PlayerCardPick),
    ReserveCard(CardPickOnBoard),
    #[allow(dead_code)]
    Noop, // reserved for testing, player passes their turn
}

pub trait GameTurn<T: PlayerScopedGameData> {
    fn take_turn(&self, game: &mut T) -> Result<TurnSuccess, TurnFailed>;
    /// Perform any validation that can be applied to self-data alone
    fn is_valid(&self) -> bool;
    fn can_take_turn(&self, game: &T) -> bool;
}

#[derive(Debug, PartialEq)]
pub enum TurnPlanningFailed{
    #[allow(dead_code)]
    UnknownError,
    MissingPlayer,
    CantTakeTokens,
    CantSpendTokens,
    MissingCard,
    PurchaseOtherPlayerReservedCard,
    CantTakeCard,
}

impl Turn {
    pub(crate) fn get_sub_turns<T: PlayerScopedGameData>
        (&self, game: &T) -> Result<Vec<SubTurn>, TurnPlanningFailed> {
        use crate::game_actions::turn::TurnPlanningFailed::*;
        
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let take_tokens = SubTurnAction::TransactTokens(
                    get_transaction_sequence(-1, &[*a, *b, *c])
                ).to_partial();
                if !take_tokens.can_complete(game) {
                    return Err(CantTakeTokens);
                }
                Ok(vec![
                    take_tokens
                ])
            },

            Turn::TakeTwoTokens(a) => {
                let take_tokens = SubTurnAction::TransactTokens(
                    get_transaction_sequence(-1, &[*a, *a])
                ).to_partial();
                if !take_tokens.can_complete(game) {
                    return Err(CantTakeTokens);
                }
                Ok(vec![
                    take_tokens
                ])
            }
            Turn::PurchaseCard(card_pick) => {
                let Some(picked_card) = game.get_card_pick(card_pick) else {
                    return Err(MissingCard)
                };

                let mut result_actions = vec![];

                let base_cost = picked_card.cost;
                let resources_from_cards = game.persistent_resources();
                let mut modified_cost = base_cost.clone();
                for resource in ResourceType::iterator() {
                    modified_cost[*resource] = max(0, modified_cost[*resource] - resources_from_cards[*resource]);
                }

                let mut bank_transactions = vec![];
                let owned_tokens = game.owned_resources();
                let mut gold_deficit = 0;
                for resource in ResourceType::iterator() {
                    let cost = modified_cost[*resource];
                    if cost <= 0{
                        continue;
                    }
                    let spent_tokens = min(cost, owned_tokens[*resource]);
                    if spent_tokens > 0{
                        bank_transactions.push(BankTransaction{
                            resource: CostType(*resource),
                            amount: spent_tokens
                        });
                    }

                    let deficit = cost - spent_tokens;
                    gold_deficit += deficit;
                }
                if gold_deficit > 0 {
                    bank_transactions.push(BankTransaction{
                        resource: ResourceTokenType::Gold,
                        amount: gold_deficit
                    });
                }

                if bank_transactions.len() > 0 {
                    let take_tokens = SubTurnAction::TransactTokens(bank_transactions).to_required();
                    if !take_tokens.can_complete(game) {
                        return Err(CantSpendTokens)
                    }
                    result_actions.push(take_tokens);
                }

                let selection_type = match *card_pick {
                    PlayerCardPick::OnBoard(x) => CardSelectionType::ObtainBoard(x),
                    PlayerCardPick::Reserved(x) => CardSelectionType::ObtainReserved(x),
                };

                let take_card = SubTurnAction::TransactCard(CardTransaction{
                    selection_type
                }).to_required();
                if !take_card.can_complete(game) {
                    return Err(CantTakeCard);
                }
                result_actions.push(take_card);
                Ok(result_actions)
            }
            Turn::ReserveCard(reserved_card) => {
                let reserve_card = SubTurnAction::TransactCard(CardTransaction{
                    selection_type: CardSelectionType::Reserve(*reserved_card)
                });

                let take_gold = SubTurnAction::TransactTokens(vec![
                    BankTransaction{resource: ResourceTokenType::Gold, amount: -1},
                ]);

                Ok(vec![
                    reserve_card.to_required(),
                    take_gold.to_partial()
                ])
            }
            Turn::Noop => Ok(vec![])
        }
    }
}

impl<T: PlayerScopedGameData> GameTurn<T> for Turn {
    fn take_turn(&self, game: &mut T) -> Result<TurnSuccess, TurnFailed> {

        let Ok(sub_turns) = self.get_sub_turns(game) else {
            return Err(TurnFailed::FailureNoModification)
        };

        let mut is_first_turn = true;
        let mut success_mode: Option<TurnSuccess> = None;
        for sub_turn in sub_turns {
            let sub_result = sub_turn.action
                .do_sub_turn(game)
                .map_err(|e| if is_first_turn { e } else {FailurePartialModification})?;

            let may_continue = match sub_result {
                TurnSuccess::Success => true,
                TurnSuccess::SuccessPartial => sub_turn.failure_mode == MayPartialSucceed,
            };

            if !may_continue {
                return Err(FailurePartialModification);
            }

            success_mode = Some(match success_mode {
                None => sub_result,
                Some(last_success) => last_success.combine(&sub_result)
            });

            is_first_turn = false;
        }

        match success_mode {
            None => Err(FailurePartialModification),
            Some(x) => Ok(x)
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

    fn can_take_turn(&self, game: &T) -> bool {
        // ??? why do I need to do this? is_valid() is *Literally* right there, why can't I just call it?
        let is_valid = <Turn as GameTurn<T>>::is_valid(self);
        is_valid && self.get_sub_turns(game).is_ok()
    }
}