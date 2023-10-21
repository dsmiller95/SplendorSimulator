use seq_macro::seq;
use crate::constants::{MAX_PLAYER_COUNT, CardPickOnBoard, ResourceTokenBank, ResourceAmountFlags, ReservedCardSelection, CardPickInReservedCards, reserved_card, MAX_INVENTORY_TOKENS};
use crate::constants::CardPickInTier::OpenCard;
use crate::constants::CardTier::{CardTier1, CardTier2};
use crate::constants::GlobalCardPick::OnBoard;
use crate::constants::OpenCardPickInTier::{OpenCardPickInTier1, OpenCardPickInTier2};
use crate::constants::PlayerSelection::*;
use crate::constants::ReservedCardSelection::*;
use crate::constants::ResourceTokenType::Gold;
use crate::constants::ResourceType::*;
use crate::game_actions::knowable_game_data::{HasCards, KnowableActorData, KnowableGameData};
use crate::game_actions::turn::{GameTurn, Turn, TurnFailed, TurnSuccess};
use crate::game_model::game_components::Card;
use crate::game_model::game_full::GameModel;



fn get_test_game() -> GameModel {
    let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
    game.bank_resources = [5, 5, 5, 5, 5, 6];
    
    game
}

fn test_purchase_result(
    bank: ResourceTokenBank,
    player_bank: ResourceTokenBank,
    player_persistent: ResourceAmountFlags,
    card: Card,
    expected_result: Result<TurnSuccess, TurnFailed>,
    expected_bank: ResourceTokenBank,
    expected_player_bank: ResourceTokenBank){
    
    let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
    game.bank_resources = bank;
    let actor =game.actors[PlayerSelection2].as_mut().unwrap();
    actor.resource_tokens = player_bank;
    actor.resources_from_cards = player_persistent;

    game.card_rows_sized[CardTier1].open_cards[OpenCardPickInTier2] = Some(card);
    
    let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard {
        tier: CardTier1,
        pick: OpenCard(OpenCardPickInTier2),
    }));

    assert_eq!(turn.can_take_turn(&game, PlayerSelection2), true);
    let turn_result = turn.take_turn(&mut game, PlayerSelection2);
    assert_eq!(turn_result, expected_result);
    if expected_result == Ok(TurnSuccess::Success) {
        assert_eq!(game.bank_resources, expected_bank);
        assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, expected_player_bank);
    }
    else{
        assert_eq!(game.bank_resources, bank);
        assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, player_bank);
    }
}

#[test]
fn does_reserve_card_from_board() {
    let player_n = PlayerSelection2;
    let card_pick = CardPickOnBoard {
        tier: CardTier1,
        pick: OpenCard(OpenCardPickInTier2),
    };

    let card_id = 24;
    let card = Card::new().with_id(card_id);

    let mut game = get_test_game();
    let mut sized = game;
    sized.try_put_card(&card_pick.into(), card).unwrap();
    let actor = sized.get_actor_at_index(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 0);

    let turn = Turn::ReserveCard(card_pick);
    assert_eq!(turn.can_take_turn(&sized, player_n), true);
    let turn_result = turn.take_turn(&mut sized, player_n);
    assert_eq!(turn_result, Ok(TurnSuccess::Success));

    let actor = sized.get_actor_at_index(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 1);
    assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
    assert_eq!(actor.resource_tokens[Gold], 1);
}


#[test]
fn does_reserve_card_from_board_when_full_token_inventory() {
    let player_n = PlayerSelection2;
    let card_pick = CardPickOnBoard {
        tier: CardTier1,
        pick: OpenCard(OpenCardPickInTier2),
    };

    let card_id = 24;
    let card = Card::new().with_id(card_id);

    let mut game = get_test_game();
    let mut sized = game;
    
    sized.try_put_card(&card_pick.into(), card).unwrap();
    let actor = sized.get_actor_at_index_mut(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 0);
    actor.resource_tokens[Gold] = MAX_INVENTORY_TOKENS;

    let turn = Turn::ReserveCard(card_pick);
    assert_eq!(turn.can_take_turn(&sized, player_n), true);
    let turn_result = turn.take_turn(&mut sized, player_n);
    assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));

    let actor = sized.get_actor_at_index(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 1);
    assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
    assert_eq!(actor.resource_tokens[Gold], 10);
}

#[test]
fn can_not_reserve_card_when_full_reservations() {
    let player_n = PlayerSelection2;
    let card_pick = CardPickOnBoard {
        tier: CardTier2,
        pick: OpenCard(OpenCardPickInTier1),
    };

    let card1 = Card::new().with_id(1);
    let card2 = Card::new().with_id(2);
    let card3 = Card::new().with_id(3);
    let card4 = Card::new().with_id(4);
    
    let mut game = get_test_game();
    let mut sized = game;
    sized.try_put_card(&card_pick.into(), card1).unwrap();
    
    for (a, b) in [
        (ReservedCardSelection1, card2),
        (ReservedCardSelection2, card3),
        (ReservedCardSelection3, card4),
    ] {
        sized.try_put_card(&reserved_card(player_n, a), b).unwrap();
    }
    
    let actor = sized.get_actor_at_index(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 3);

    let turn = Turn::ReserveCard(card_pick);
    let turn_result = turn.take_turn(&mut sized, player_n);
    assert_eq!(turn_result, Err(TurnFailed::FailureNoModification));
    let actor = sized.get_actor_at_index(player_n).unwrap();
    assert_eq!(actor.iterate_reserved_cards().count(), 3);
    assert_eq!(actor.resource_tokens[Gold], 0);
}