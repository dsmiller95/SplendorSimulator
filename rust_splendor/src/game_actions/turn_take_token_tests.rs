#[cfg(test)]
mod tests {
    use crate::constants::{MAX_INVENTORY_TOKENS, ResourceTokenType, MAX_PLAYER_COUNT};
    use crate::constants::PlayerSelection::*;
    use crate::constants::ResourceType::*;
    use crate::game_actions::turn::{GameTurn, Turn};
    use crate::game_actions::turn_result::TurnSuccess;


    #[test]
    fn cannot_act_on_missing_player() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 10;
        let turn = Turn::TakeTwoTokens(Diamond);
        assert_eq!(turn.can_take_turn(&game, PlayerSelection3), false);
    }

    #[test]
    fn cannot_take_three_of_same_token() {
        let game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        let turn = Turn::TakeThreeTokens(Diamond, Diamond, Emerald);
        assert_eq!(turn.can_take_turn(&game, PlayerSelection2), false);
    }

    #[test]
    fn take_two_takes_one_token_when_one_in_bank_valid() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Diamond] = 1;
        let turn = Turn::TakeTwoTokens(Diamond);
        assert_eq!(turn.can_take_turn(&game, PlayerSelection2), true);
    }

    #[test]
    fn take_two_takes_one_token_when_one_in_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Diamond] = 1;
        let turn = Turn::TakeTwoTokens(Diamond);
        let turn_result = turn.take_turn(&mut game, PlayerSelection2);
        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));
        assert_eq!(game.bank_resources[Diamond], 0);
        assert_eq!(game.actors[PlayerSelection2].as_ref().unwrap().resource_tokens[Diamond], 1);
    }


    #[test]
    fn take_three_takes_two_tokens_when_two_in_bank_is_valid() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 0;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);
        assert_eq!(turn.can_take_turn(&game, PlayerSelection1), true);
    }

    #[test]
    fn take_three_takes_two_tokens_when_two_in_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 0;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);
        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_two_tokens_when_two_in_bank_of_requested() {
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
        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_two_tokens_based_on_ordering_when_capacity_max() {
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


        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_two_tokens_based_on_ordering_when_capacity_max_different_order() {
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

        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_two_tokens_based_on_ordering_when_capacity_max_skips_empty_bank() {
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

        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_three_tokens_when_three_in_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 1;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);
        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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
    fn take_three_takes_three_tokens_when_three_in_bank_valid() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 1;
        game.bank_resources[Diamond] = 1;
        game.bank_resources[Emerald] = 1;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);
        assert_eq!(turn.can_take_turn(&game, PlayerSelection1), true);
    }

    #[test]
    fn take_three_takes_three_tokens_when_many_in_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources[Ruby] = 10;
        game.bank_resources[Diamond] = 13;
        game.bank_resources[Emerald] = 11;
        let turn = Turn::TakeThreeTokens(
            Ruby,
            Diamond,
            Emerald);
        let turn_result = turn.take_turn(&mut game, PlayerSelection1);
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