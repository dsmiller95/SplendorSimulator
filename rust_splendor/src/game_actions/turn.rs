use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::constants::{CardPickOnBoard, GlobalCardPick, PlayerSelection, ResourceType, ResourceTokenType};
use crate::game_actions::bank_transactions::{BankTransaction, can_transact};
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


impl  Turn {
    fn get_sub_turns<T: KnowableGameData<ActorType>, ActorType : KnowableActorData>
        (&self, game: &mut T, actor_index: PlayerSelection) -> Option<Vec<SubTurn>> {

        if !self.can_take_turn(game, actor_index) {
            return None;
        }

        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let take_tokens = SubTurnAction::TransactTokens(vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*b), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*c), amount: -1},
                ]);
                Some(vec![SubTurn{
                    action: take_tokens,
                    failure_mode: SubTurnFailureMode::MayPartialSucceed
                }])
            },

            Turn::TakeTwoTokens(a) => {
                let take_tokens = SubTurnAction::TransactTokens(vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                    BankTransaction{player: actor_index, resource: ResourceTokenType::CostType(*a), amount: -1},
                ]);
                Some(vec![SubTurn{
                    action: take_tokens,
                    failure_mode: SubTurnFailureMode::MayPartialSucceed
                }])
            }
            Turn::PurchaseCard(_) => {
                todo!()
            }
            Turn::ReserveCard(reserved_card) => {
                let reserve_card = SubTurnAction::TransactCard(CardTransaction{
                    player: actor_index,
                    selection_type: CardSelectionType::Reserve(*reserved_card)
                });


                let take_gold = SubTurnAction::TransactTokens(vec![
                    BankTransaction{player: actor_index, resource: ResourceTokenType::Gold, amount: -1},
                ]);

                Some(vec![
                    SubTurn{
                        action: reserve_card,
                        failure_mode: SubTurnFailureMode::MustAllSucceed
                    },
                    SubTurn{
                        action: take_gold,
                        failure_mode: SubTurnFailureMode::MayPartialSucceed
                    }
                ])
            }
            Turn::Noop => Some(vec![])
        }
    }
}

impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> GameTurn<T, ActorType> for Turn {
    fn take_turn(&self, game: &mut T, actor_index: PlayerSelection) -> Result<TurnSuccess, TurnFailed> {

        let Some(sub_turns) = self.get_sub_turns(game, actor_index) else {
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