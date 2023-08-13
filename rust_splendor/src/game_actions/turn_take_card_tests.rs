use crate::constants::{MAX_PLAYER_COUNT, CardPickOnBoard, ResourceTokenBank};
use crate::constants::CardPickInTier::OpenCard;
use crate::constants::CardTier::CardTier1;
use crate::constants::GlobalCardPick::OnBoard;
use crate::constants::OpenCardPickInTier::OpenCardPickInTier2;
use crate::constants::PlayerSelection::*;
use crate::constants::ResourceType::*;
use crate::game_actions::turn::{GameTurn, Turn, TurnResult};
use crate::game_model::game_components::Card;


#[test]
fn cannot_purchase_empty_card() {
    let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
    game.game_sized.card_rows[CardTier1].open_cards[OpenCardPickInTier2] = None;

    let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard {
        tier: CardTier1,
        pick: OpenCard(OpenCardPickInTier2),
    }));

    assert_eq!(turn.can_take_turn(&game.game_sized, PlayerSelection2), false);
}

fn test_purchase_result(
    bank: ResourceTokenBank,
    player_bank: ResourceTokenBank,
    card: Card,
    expected_result: TurnResult,
    expected_bank: ResourceTokenBank,
    expected_player_bank: ResourceTokenBank){
    
    let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
    game.game_sized.bank_resources = bank;
    game.game_sized.actors[PlayerSelection2].as_mut().unwrap().resource_tokens = player_bank;

    game.game_sized.card_rows[CardTier1].open_cards[OpenCardPickInTier2] = Some(card);
    
    let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard {
        tier: CardTier1,
        pick: OpenCard(OpenCardPickInTier2),
    }));

    assert_eq!(turn.can_take_turn(&game.game_sized, PlayerSelection2), true);
    let turn_result = turn.take_turn(&mut game.game_sized, PlayerSelection2);
    assert_eq!(turn_result, expected_result);
    if expected_result == TurnResult::Success {
        assert_eq!(game.game_sized.bank_resources, expected_bank);
        assert_eq!(game.game_sized.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, expected_player_bank);
    }
    else{
        assert_eq!(game.game_sized.bank_resources, bank);
        assert_eq!(game.game_sized.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, player_bank);
    }
}

#[test]
fn cannot_purchase_expensive_card() {
    test_purchase_result(
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        Card::new([1, 0, 1, 0, 0]),
        TurnResult::FailureNoModification,
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    );
}
#[test]
fn can_purchase_card_with_all_resources() {
    test_purchase_result(
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0],
        Card::new([1, 0, 1, 0, 0]),
        TurnResult::FailureNoModification,
        [2, 1, 2, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    );
}
#[test]
fn can_purchase_card_with_gold() {
    test_purchase_result(
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        Card::new([1, 0, 1, 0, 0]),
        TurnResult::Success,
        [2, 1, 1, 1, 1, 2],
        [0, 0, 0, 0, 0, 0],
    );
}

#[test]
fn avoids_purchase_card_with_gold() {
    test_purchase_result(
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1],
        Card::new([1, 0, 1, 0, 0]),
        TurnResult::Success,
        [2, 1, 2, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    );
}
