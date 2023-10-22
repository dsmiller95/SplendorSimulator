#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::constants::{MAX_PLAYER_COUNT, CardPickOnBoard, ResourceTokenBank, ResourceAmountFlags, board_card};

    use crate::constants::CardPickInTier::OpenCard;
    use crate::constants::CardTier::CardTier1;
    use crate::constants::GlobalCardPick::OnBoard;
    use crate::constants::OpenCardPickInTier::OpenCardPickInTier2;
    use crate::constants::PlayerSelection::*;
    use crate::constants::ResourceTokenType::{CostType, Gold};
    use crate::constants::ResourceType::*;
    use crate::game_actions::bank_transactions::{BankTransaction, get_transaction_sequence, get_transaction_sequence_tokens};
    use crate::game_actions::card_transaction::CardSelectionType::ObtainBoard;
    use crate::game_actions::card_transaction::CardTransaction;
    use crate::game_actions::player_scoped_game_data::CanPlayerScope;
    use crate::game_actions::sub_turn::{SubTurn};
    use crate::game_actions::sub_turn::SubTurnAction::{TransactCard, TransactTokens};
    use crate::game_actions::test_utils::get_test_game;
    use crate::game_actions::turn::{Turn, TurnPlanningFailed};
    use crate::game_actions::turn::TurnPlanningFailed::*;
    use crate::game_model::game_components::Card;

    #[test]
    fn when_empty_card__produce_none() {
        // arrange
        let mut game = get_test_game(MAX_PLAYER_COUNT);
        game.card_rows_sized[CardTier1].open_cards[OpenCardPickInTier2] = None;

        let turn = Turn::PurchaseCard(OnBoard(CardPickOnBoard {
            tier: CardTier1,
            pick: OpenCard(OpenCardPickInTier2),
        }));

        // act
        let mut scoped = game.scope_to(PlayerSelection2).unwrap();
        let sub_turns = turn.get_sub_turns(&scoped, PlayerSelection2);

        // assert
        assert_eq!(sub_turns, Err(MissingCard));
    }

    fn default_card_transact() -> SubTurn {
        TransactCard(CardTransaction{
            player: PlayerSelection2,
            selection_type: ObtainBoard(CardPickOnBoard{
                tier: CardTier1,
                pick: OpenCard(OpenCardPickInTier2),
            })
        }).to_required()
    }

    fn test_sub_turn_result(
        bank: ResourceTokenBank,
        player_bank: ResourceTokenBank,
        player_persistent: ResourceAmountFlags,
        card: Card,
        expected_result: Result<Vec<SubTurn>, TurnPlanningFailed>
    ){
        let mut game = get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources = bank;
        let actor = game.actors[PlayerSelection2].as_mut().unwrap();
        actor.resource_tokens = player_bank;
        actor.resources_from_cards = player_persistent;

        game.card_rows_sized[CardTier1].open_cards[OpenCardPickInTier2] = Some(card);

        let turn = Turn::PurchaseCard(board_card(CardTier1, OpenCard(OpenCardPickInTier2)));

        // act
        let mut scoped = game.scope_to(PlayerSelection2).unwrap();
        let sub_turns = turn.get_sub_turns(&scoped, PlayerSelection2);

        // assert
        assert_eq!(sub_turns, expected_result);
    }

    #[test]
    fn when_available_resources_exact__produces_bank_transactions() {
        let bank_transact =
            TransactTokens(
                get_transaction_sequence(
                    PlayerSelection2,
                    1,
                    &[Ruby, Sapphire])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }
    #[test]
    fn when_multiple_tokens_of_type_required__produces_single_bank_transaction(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence(
                    PlayerSelection2,
                    2,
                    &[Emerald])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([0, 2, 0, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }

    #[test]
    fn when_available_resources_excess__produces_bank_transactions(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence(
                    PlayerSelection2,
                    1,
                    &[Ruby, Sapphire])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [2, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }

    #[test]
    fn when_available_resources_from_persistent__produces_no_bank_transactions(){
        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![default_card_transact()])
        )
    }

    #[test]
    fn when_available_resources_from_partial_persistent__produces_partial_bank_transaction(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence(
                    PlayerSelection2,
                    1,
                    &[Sapphire])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }

    #[test]
    fn when_available_resources_from_gold__produces_partial_gold_bank_transaction(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence_tokens(
                    PlayerSelection2,
                    1,
                    &[CostType(Ruby), Gold])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }
    #[test]
    fn when_available_resources_from_gold__produces_complete_gold_bank_transaction(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence_tokens(
                    PlayerSelection2,
                    2,
                    &[Gold])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }
    #[test]
    fn when_available_resources_and_excess_gold__produces_non_gold_bank_transactions(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence(
                    PlayerSelection2,
                    1,
                    &[Sapphire, Diamond])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([0, 0, 1, 1, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }
    #[test]
    fn when_multiple_tokens_of_type_required_and_gold_available__produces_mixed_gold_bank_transaction(){
        let bank_transact =
            TransactTokens(
                get_transaction_sequence_tokens(
                    PlayerSelection2,
                    2,
                    &[CostType(Emerald), Gold])
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 2, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([0, 4, 0, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }
    #[test]
    fn when_multiple_tokens_of_type_required_and_gold_available__produces_mixed_gold_bank_transaction_with_heterogeneous_amounts(){
        let bank_transact =
            TransactTokens(
                vec![
                    BankTransaction{
                        player: PlayerSelection2,
                        resource: CostType(Emerald),
                        amount: 1
                    },
                    BankTransaction{
                        player: PlayerSelection2,
                        resource: Gold,
                        amount: 2
                    },
                ]
            ).to_required();

        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([0, 3, 0, 0, 0]),
            Ok(vec![bank_transact, default_card_transact()])
        )
    }

    #[test]
    fn when_missing_resources__cannot_spend_tokens(){
        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Err(CantSpendTokens)
        )
    }

    #[test]
    fn when_missing_resources_partial__cannot_spend_tokens(){
        test_sub_turn_result(
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            Card::new().with_cost([1, 0, 1, 0, 0]),
            Err(CantSpendTokens)
        )
    }
}