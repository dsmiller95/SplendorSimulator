#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::constants::{MAX_PLAYER_COUNT, CardPickOnBoard, ResourceTokenBank, ResourceAmountFlags};

    use crate::constants::CardPickInTier::OpenCard;
    use crate::constants::CardTier::CardTier1;
    use crate::constants::PlayerCardPick::OnBoard;
    use crate::constants::OpenCardPickInTier::OpenCardPickInTier2;
    use crate::constants::PlayerSelection::*;
    use crate::game_actions::player_scoped_game_data::CanPlayerScope;
    use crate::game_actions::turn::{GameTurn, Turn};
    use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};
    use crate::game_model::game_components::Card;

    #[test]
    fn cannot_purchase_empty_card() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.card_rows[CardTier1].open_cards[OpenCardPickInTier2] = None;

        let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard {
            tier: CardTier1,
            pick: OpenCard(OpenCardPickInTier2),
        }));

        let (_, turn_result) = game.on_player(PlayerSelection2, |scoped| {
            turn.can_take_turn(scoped)
        });

        assert_eq!(turn_result, false);
    }

    fn test_purchase_result(
        bank: ResourceTokenBank,
        player_bank: ResourceTokenBank,
        player_persistent: ResourceAmountFlags,
        card: Card,
        expected_result: Result<TurnSuccess, TurnFailed>,
        expected_bank: ResourceTokenBank,
        expected_player_bank: ResourceTokenBank) {

        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources = bank;
        let actor = game.actors[PlayerSelection2].as_mut().unwrap();
        actor.resource_tokens = player_bank;
        actor.resources_from_cards = player_persistent;

        game.card_rows[CardTier1].open_cards[OpenCardPickInTier2] = Some(card);

        let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard{
            tier: CardTier1,
            pick: OpenCard(OpenCardPickInTier2)
        }));

        let (game, turn_result) = game.on_player(PlayerSelection2, |scoped| {
            turn.take_turn(scoped)
        });

        assert_eq!(turn_result, expected_result);
        if expected_result == Ok(TurnSuccess::Success) {
            assert_eq!(game.bank_resources, expected_bank);
            assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, expected_player_bank);
        } else {
            assert_eq!(game.bank_resources, bank);
            assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens, player_bank);
        }
    }

    #[test]
    fn when_expensive_card__cannot_purchase() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Err(TurnFailed::FailureNoModification),
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        );
    }

    #[test]
    fn when_affordable_card_exact__purchases() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(TurnSuccess::Success),
            [2, 1, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        );
    }

    #[test]
    fn when_affordable_card_with_gold__purchases() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(TurnSuccess::Success),
            [2, 1, 1, 1, 1, 2],
            [0, 0, 0, 0, 0, 0],
        );
    }

    #[test]
    fn when_affordable_card_with_persistent__purchases() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 4, 4, 0, 10],
            Card::new().with_cost([0, 1, 1, 0, 7]),
            Ok(TurnSuccess::Success),
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        );
    }

    #[test]
    fn when_expensive_card_with_not_enough_persistent__cannot_purchase() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 4, 4, 0, 5],
            Card::new().with_cost([0, 1, 1, 0, 7]),
            Err(TurnFailed::FailureNoModification),
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        );
    }

    #[test]
    fn when_affordable_card_with_combined_gold_and_persistent__purchases() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 3],
            [0, 4, 4, 0, 5],
            Card::new().with_cost([0, 1, 1, 0, 7]),
            Ok(TurnSuccess::Success),
            [1, 1, 1, 1, 1, 3],
            [0, 0, 0, 0, 0, 1],
        );
    }

    #[test]
    fn when_affordable_card_with_extra_gold__purchases_without_gold() {
        test_purchase_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(TurnSuccess::Success),
            [2, 1, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 1],
        );
    }
}