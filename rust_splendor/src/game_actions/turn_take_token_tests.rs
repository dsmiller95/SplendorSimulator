#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use std::panic;
    use crate::constants::{MAX_INVENTORY_TOKENS, ResourceTokenType, MAX_PLAYER_COUNT};
    use crate::constants::PlayerSelection::*;
    use crate::constants::ResourceType::*;
    use crate::game_actions::player_scoped_game_data::CanPlayerScope;
    use crate::game_actions::turn::{GameTurn, Turn};
    use crate::game_actions::turn_result::TurnSuccess;

    #[test]
    fn when_player_missing__cannot_take_tokens() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 10;
        let turn = Turn::TakeTwoTokens(Diamond);

        let panic_result = panic::catch_unwind(|| {
            let (game, can_turn) = game.on_player(PlayerSelection3, |scoped| {
                turn.can_take_turn(scoped, PlayerSelection3)
            });
        });

        assert!(panic_result.is_err());
    }

    #[test]
    fn when_taking_three_of_same_type__cannot_take() {
        let game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        let turn = Turn::TakeThreeTokens(Diamond, Diamond, Emerald);

        let (game, can_turn) = game.on_player(PlayerSelection2, |scoped| {
            turn.can_take_turn(scoped, PlayerSelection2)
        });

        assert_eq!(can_turn, false);
    }

    #[test]
    fn when_taking_two_and_bank_has_one__takes_only_one() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Diamond] = 1;
        let turn = Turn::TakeTwoTokens(Diamond);

        let (game, turn_result) = game.on_player(PlayerSelection2, |scoped| {
            turn.take_turn(scoped, PlayerSelection2)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        assert_eq!(game.bank_resources[Diamond], 0);
        assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens[Diamond], 1);
    }

    #[test]
    fn when_taking_three_and_bank_has_two__takes_only_two() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 0;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Ruby], 0);
        assert_eq!(game_bank[Diamond], 0);
        assert_eq!(game_bank[Emerald], 0);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Ruby], 0);
        assert_eq!(player_bank[Diamond], 1);
        assert_eq!(player_bank[Emerald], 1);
    }

    #[test]
    fn when_taking_three_and_bank_has_two_and_others__takes_only_two() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 0;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;

        game.bank_resources[Sapphire] = 10;
        game.bank_resources[Onyx] = 10;
        game.bank_resources[ResourceTokenType::Gold] = 10;

        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Ruby], 0);
        assert_eq!(game_bank[Diamond], 0);
        assert_eq!(game_bank[Emerald], 0);
        assert_eq!(game_bank[Sapphire], 10);
        assert_eq!(game_bank[Onyx], 10);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Ruby], 0);
        assert_eq!(player_bank[Diamond], 1);
        assert_eq!(player_bank[Emerald], 1);
    }


    #[test]
    fn when_inventory_nearly_full__takes_first_tokens_only() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 1;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;

        game.actors[PlayerSelection1].as_mut().unwrap()
            .resource_tokens[Onyx] = MAX_INVENTORY_TOKENS - 2;

        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Ruby], 0);
        assert_eq!(game_bank[Diamond], 0);
        assert_eq!(game_bank[Emerald], 1);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Ruby], 1);
        assert_eq!(player_bank[Diamond], 1);
        assert_eq!(player_bank[Emerald], 0);
    }

    // TODO: in the previous model, we used a discard-down meta-turn action to allow the player to choose
    //  to discard a different token than one they picked. how can we test this, and represent it, in the turn?
    //  it may be that we dont need to pay attention to MAX_INVENTORY_TOKENS in this phase...
    #[test]
    fn when_inventory_nearly_full__takes_first_tokens_only_in_different_order() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 1;
        game.bank_resources[Emerald] = 1;
        game.bank_resources[Sapphire] = 1;

        game.actors[PlayerSelection1].as_mut().unwrap()
            .resource_tokens[Onyx] = MAX_INVENTORY_TOKENS - 2;

        let turn = Turn::TakeThreeTokens(
            Emerald,
            Sapphire,
            Ruby);


        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Emerald], 0);
        assert_eq!(game_bank[Sapphire], 0);
        assert_eq!(game_bank[Ruby], 1);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Emerald], 1);
        assert_eq!(player_bank[Sapphire], 1);
        assert_eq!(player_bank[Ruby], 0);
    }

    #[test]
    fn when_inventory_nearly_full_and_bank_partially_empty__takes_tokens_in_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Emerald] = 1;
        game.bank_resources[Sapphire] = 0;
        game.bank_resources[Ruby] = 1;

        game.actors[PlayerSelection1].as_mut().unwrap()
            .resource_tokens[Onyx] = MAX_INVENTORY_TOKENS - 2;

        let turn = Turn::TakeThreeTokens(
            Emerald,
            Sapphire,
            Ruby);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Emerald], 0);
        assert_eq!(game_bank[Sapphire], 0);
        assert_eq!(game_bank[Ruby], 0);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Emerald], 1);
        assert_eq!(player_bank[Sapphire], 0);
        assert_eq!(player_bank[Ruby], 1);
    }

    #[test]
    fn when_taking_three_exact_in_bank__takes_three() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 1;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::Success));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Ruby], 0);
        assert_eq!(game_bank[Diamond], 0);
        assert_eq!(game_bank[Emerald], 0);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Ruby], 1);
        assert_eq!(player_bank[Diamond], 1);
        assert_eq!(player_bank[Emerald], 1);
    }

    #[test]
    fn when_taking_three_excess_in_bank__takes_three() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 10;
        game.bank_resources[Diamond] = 13;
        game.bank_resources[Emerald] = 11;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);

        let (game, turn_result) = game.on_player(PlayerSelection1, |scoped| {
            turn.take_turn(scoped, PlayerSelection1)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::Success));
        let game_bank = game.bank_resources;
        assert_eq!(game_bank[Ruby], 9);
        assert_eq!(game_bank[Diamond], 12);
        assert_eq!(game_bank[Emerald], 10);
        let player_bank = game.actors[PlayerSelection1].as_ref().unwrap().resource_tokens;
        assert_eq!(player_bank[Ruby], 1);
        assert_eq!(player_bank[Diamond], 1);
        assert_eq!(player_bank[Emerald], 1);
    }
}