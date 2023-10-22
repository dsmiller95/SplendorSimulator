#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::constants::{MAX_PLAYER_COUNT, CardPickOnBoard, reserved_card, MAX_INVENTORY_TOKENS};
    use crate::constants::CardPickInTier::OpenCard;
    use crate::constants::CardTier::{CardTier1, CardTier2};
    use crate::constants::OpenCardPickInTier::{OpenCardPickInTier1, OpenCardPickInTier2};
    use crate::constants::PlayerSelection::*;
    use crate::constants::ReservedCardSelection::*;
    use crate::constants::ResourceTokenType::Gold;
    use crate::game_actions::knowable_game_data::{HasCards, KnowableActorData, KnowableGameData};
    use crate::game_actions::player_scoped_game_data::CanPlayerScope;
    use crate::game_actions::turn::{GameTurn, Turn};
    use crate::game_actions::turn_result::{TurnFailed, TurnSuccess};
    use crate::game_model::game_components::Card;
    use crate::game_model::game_full::GameModel;


    fn get_test_game() -> GameModel {
        let mut game = crate::game_actions::test_utils::get_test_game(MAX_PLAYER_COUNT);
        game.bank_resources = [5, 5, 5, 5, 5, 6];

        game
    }
    #[test]
    fn when_card_on_board__reserves_card() {
        let player_n = PlayerSelection2;
        let card_pick = CardPickOnBoard {
            tier: CardTier1,
            pick: OpenCard(OpenCardPickInTier2),
        };

        let card_id = 24;
        let card = Card::new().with_id(card_id);

        let mut game = get_test_game();
        game.try_put_card(&card_pick.into(), card).unwrap();
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 0);

        let turn = Turn::ReserveCard(card_pick);
        let (game, turn_result) = game.on_player(player_n, |scoped| {
            turn.take_turn(scoped, player_n)
        });

        assert_eq!(turn_result, Ok(TurnSuccess::Success));

        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);
        assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
        assert_eq!(actor.resource_tokens[Gold], 1);
    }


    #[test]
    fn when_player_tokens_full__reserves_card_and_wastes_gold() {
        let player_n = PlayerSelection2;
        let card_pick = CardPickOnBoard {
            tier: CardTier1,
            pick: OpenCard(OpenCardPickInTier2),
        };

        let card_id = 24;
        let card = Card::new().with_id(card_id);

        let mut game = get_test_game();

        game.try_put_card(&card_pick.into(), card).unwrap();
        let actor = game.get_actor_at_index_mut(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 0);
        actor.resource_tokens[Gold] = MAX_INVENTORY_TOKENS;

        let turn = Turn::ReserveCard(card_pick);
        let (game, turn_result) = game.on_player(player_n, |scoped| {
            turn.take_turn(scoped, player_n)
        });
        assert_eq!(turn_result, Ok(TurnSuccess::SuccessPartial));

        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 1);
        assert_eq!(actor.iterate_reserved_cards().next().unwrap().id, card_id);
        assert_eq!(actor.resource_tokens[Gold], 10);
    }

    #[test]
    fn when_player_reservations_full__does_not_reserve_card() {
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
        game.try_put_card(&card_pick.into(), card1).unwrap();

        for (a, b) in [
            (ReservedCardSelection1, card2),
            (ReservedCardSelection2, card3),
            (ReservedCardSelection3, card4),
        ] {
            game.try_put_card(&reserved_card(player_n, a), b).unwrap();
        }

        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 3);

        let turn = Turn::ReserveCard(card_pick);
        let (game, turn_result) = game.on_player(player_n, |scoped| {
            turn.take_turn(scoped, player_n)
        });
        assert_eq!(turn_result, Err(TurnFailed::FailureNoModification));
        let actor = game.get_actor_at_index(player_n).unwrap();
        assert_eq!(actor.iterate_reserved_cards().count(), 3);
        assert_eq!(actor.resource_tokens[Gold], 0);
    }
}