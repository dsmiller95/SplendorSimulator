use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};
use crate::constants::{CardPickOnBoard, MAX_INVENTORY_TOKENS, ResourceTokenType, GlobalCardPick};

#[derive(Debug)]
pub enum Turn {
    TakeThreeTokens(ResourceTokenType, ResourceTokenType, ResourceTokenType),
    TakeTwoTokens(ResourceTokenType),
    PurchaseCard(GlobalCardPick),
    ReserveCard(CardPickOnBoard),
    Noop, // reserved for testing, player passes their turn
}


pub trait GameTurn<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> {
    fn take_turn(&self, game: &mut T, actor_index: usize) -> TurnResult;
    fn can_take_turn(&self, game: &T, actor_index: usize) -> bool;
}

pub enum TurnResult {
    Success,
    FailureNoModification,
    FailurePartialModification,
}

impl<T: KnowableGameData<ActorType>, ActorType : KnowableActorData> GameTurn<T, ActorType> for Turn {
    fn take_turn(&self, game: &mut T, actor_index: usize) -> TurnResult {
        if !self.can_take_turn(game, actor_index) {
            return TurnResult::FailureNoModification
        }
        
        todo!()
    }

    fn can_take_turn(&self, game: &T, actor_index: usize) -> bool {
        let actor = game.get_actor_at_index(actor_index);
        if actor.is_none() {
            return false
        }
        let actor = actor.as_ref().unwrap();
        
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let mut bank = game.bank_resources().clone();
                bank[*a] -= 1;
                bank[*b] -= 1;
                bank[*c] -= 1;
                if !bank.iter().all(|&x| x >= 0){
                    return false
                }
                
                let mut resource_tokens = actor.owned_resources().clone();
                resource_tokens[*a] += 1;
                resource_tokens[*b] += 1;
                resource_tokens[*c] += 1;
                resource_tokens.iter().sum::<i8>() <= MAX_INVENTORY_TOKENS
            },
            Turn::TakeTwoTokens(a) => {
                let mut bank = game.bank_resources().clone();
                bank[*a] -= 2;
                if !bank.iter().all(|&x| x >= 0){
                    return false
                }
                
                let mut resource_tokens = actor.owned_resources().clone();
                resource_tokens[*a] += 2;
                resource_tokens.iter().sum::<i8>() <= MAX_INVENTORY_TOKENS
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