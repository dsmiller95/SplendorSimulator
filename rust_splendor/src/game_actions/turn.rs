use crate::game_model::constants::{CardPick, MAX_INVENTORY_TOKENS, ResourceTokenType};
use crate::game_model::game_full::{GameModel, HasCards};

#[derive(Debug)]
pub enum Turn {
    TakeThreeTokens(ResourceTokenType, ResourceTokenType, ResourceTokenType),
    TakeTwoTokens(ResourceTokenType),
    PurchaseCard(CardPick),
    ReserveCard(CardPick),
    Noop, // reserved for testing, player passes their turn
}


pub trait GameTurn {
    fn take_turn(&self, game: &mut GameModel, actor_index: usize) -> TurnResult;
    fn can_take_turn(&self, game: &GameModel, actor_index: usize) -> bool;
}

pub enum TurnResult {
    Success,
    FailureNoModification,
    FailurePartialModification,
}

impl GameTurn for Turn {
    fn take_turn(&self, game: &mut GameModel, actor_index: usize) -> TurnResult {
        if !self.can_take_turn(game, actor_index) {
            return TurnResult::FailureNoModification
        }
        
        todo!()
    }

    fn can_take_turn(&self, game: &GameModel, actor_index: usize) -> bool {
        let actor = &game.game_sized.actors[actor_index];
        if actor.is_none() {
            return false
        }
        let actor = actor.as_ref().unwrap();
        
        match self {
            Turn::TakeThreeTokens(a, b, c) => {
                let mut bank = game.game_sized.bank_resources.clone();
                bank[*a] -= 1;
                bank[*b] -= 1;
                bank[*c] -= 1;
                if !bank.iter().all(|&x| x >= 0){
                    return false
                }
                
                let mut resource_tokens = actor.resource_tokens.clone();
                resource_tokens[*a] += 1;
                resource_tokens[*b] += 1;
                resource_tokens[*c] += 1;
                resource_tokens.iter().sum::<i8>() <= MAX_INVENTORY_TOKENS
            },
            Turn::TakeTwoTokens(a) => {
                let mut bank = game.game_sized.bank_resources.clone();
                bank[*a] -= 2;
                if !bank.iter().all(|&x| x >= 0){
                    return false
                }
                
                let mut resource_tokens = actor.resource_tokens.clone();
                resource_tokens[*a] += 2;
                resource_tokens.iter().sum::<i8>() <= MAX_INVENTORY_TOKENS
            },
            Turn::PurchaseCard(card) => {
                let picked_card = game.game_sized.get_card_pick(card);
                if picked_card.is_none() {
                    return false
                }
                let picked_card = picked_card.unwrap();
                actor.can_afford_card(picked_card)
            },
            Turn::ReserveCard(card) => {
                let picked_card = game.game_sized.get_card_pick(card);
                
                picked_card.is_some()
                    && actor.reserved_cards.iter().any(|x| x.is_none())
            },
            Turn::Noop => true,
        }
    }
}