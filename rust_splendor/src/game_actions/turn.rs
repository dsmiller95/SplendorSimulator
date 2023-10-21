use std::cmp::{max, min};
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::constants::{CardPickOnBoard, GlobalCardPick, PlayerSelection, ResourceType, ResourceTokenType};
use crate::constants::ResourceTokenType::CostType;
use crate::game_actions::bank_transactions::{BankTransaction, can_transact, get_transaction_sequence};
use crate::game_actions::card_transaction::{CardSelectionType, CardTransaction};
use crate::game_actions::sub_turn::{SubTurn, SubTurnAction, SubTurnFailureMode};
use crate::game_actions::sub_turn::SubTurnFailureMode::MayPartialSucceed;
use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};
use crate::game_actions::turn_result::TurnFailed::FailurePartialModification;

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

#[derive(Debug, PartialEq)]
pub enum TurnPlanningFailed{
    UnknownError,
    MissingPlayer,
    CantTakeTokens,
    CantSpendTokens,
    MissingCard,
    PurchaseOtherPlayerReservedCard,
    CantTakeCard,
}

impl Turn {
    pub(crate) fn get_sub_turns<T: KnowableGameData<ActorType>, ActorType : KnowableActorData>
        (&self, game: &T, actor_index: PlayerSelection) -> Result<Vec<SubTurn>, TurnPlanningFailed> {
        use crate::game_actions::turn::TurnPlanningFailed::*;

        // if !self.can_take_turn(game, actor_index) {
        //     return Err(UnknownError);
        // }

        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let take_tokens = SubTurnAction::TransactTokens(
                    get_transaction_sequence(actor_index, -1, &[*a, *b, *c])
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
                    get_transaction_sequence(actor_index, -1, &[*a, *a])
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
                let Some(actor) = game.get_actor_at_index(actor_index) else {
                    return Err(MissingPlayer)
                };

                let mut result_actions = vec![];

                let base_cost = picked_card.cost;
                let resources_from_cards = actor.persistent_resources();
                let mut modified_cost = base_cost.clone();
                for resource in ResourceType::iterator() {
                    modified_cost[*resource] = max(0, modified_cost[*resource] - resources_from_cards[*resource]);
                }

                let mut bank_transactions = vec![];
                let owned_tokens = actor.owned_resources();
                let mut gold_deficit = 0;
                for resource in ResourceType::iterator() {
                    let cost = modified_cost[*resource];
                    if cost <= 0{
                        continue;
                    }
                    let spent_tokens = min(cost, owned_tokens[*resource]);
                    if spent_tokens > 0{
                        bank_transactions.push(BankTransaction{
                            player: actor_index,
                            resource: CostType(*resource),
                            amount: spent_tokens
                        });
                    }

                    let deficit = cost - spent_tokens;
                    gold_deficit += deficit;
                }
                if gold_deficit > 0 {
                    bank_transactions.push(BankTransaction{
                        player: actor_index,
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
                    GlobalCardPick::OnBoard(x) => CardSelectionType::ObtainBoard(x),
                    GlobalCardPick::Reserved(x) => {
                        if x.player_index != actor_index {
                            return Err(PurchaseOtherPlayerReservedCard)
                        }
                        CardSelectionType::ObtainReserved(x.reserved_card)
                    }
                };

                let take_card = SubTurnAction::TransactCard(CardTransaction{
                    player: actor_index,
                    selection_type
                }).to_required();
                if !take_card.can_complete(game){
                    return Err(CantTakeCard);
                }
                result_actions.push(take_card);
                Ok(result_actions)
            }
            Turn::ReserveCard(reserved_card) => {
                let reserve_card = SubTurnAction::TransactCard(CardTransaction{
                    player: actor_index,
                    selection_type: CardSelectionType::Reserve(*reserved_card)
                });

                let take_gold = SubTurnAction::TransactTokens(vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::Gold, amount: -1},
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

impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> GameTurn<T, ActorType> for Turn {
    fn take_turn(&self, game: &mut T, actor_index: PlayerSelection) -> Result<TurnSuccess, TurnFailed> {

        let Ok(sub_turns) = self.get_sub_turns(game, actor_index) else {
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

    fn can_take_turn(&self, game: &T, actor_index: PlayerSelection) -> bool {
        if !GameTurn::<T, ActorType>::is_valid(self) {
            return false
        }
        return self.get_sub_turns(game, actor_index).is_ok();
    }
}