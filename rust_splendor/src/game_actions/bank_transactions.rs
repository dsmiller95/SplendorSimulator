use crate::constants::{MAX_INVENTORY_TOKENS, PlayerSelection, ResourceTokenType, ResourceType};
use crate::constants::ResourceTokenType::CostType;
use crate::game_actions::knowable_game_data::{KnowableActorData, KnowableGameData};

impl BankTransaction {
    pub fn can_transact<ActorType, T>(&self, game: &T) -> Result<(), BankTransactionError>
        where ActorType: KnowableActorData,
              T: KnowableGameData<ActorType>
    {
        let bank_resources = game.bank_resources();
        let player_resources = game.get_actor_at_index(self.player)
            .ok_or(BankTransactionError::PlayerDoesNotExist)?
            .owned_resources();

        match self.amount {
            0 => return Ok(()),
            ..=-1 => {
                if bank_resources[self.resource] + self.amount < 0 {
                    return Err(BankTransactionError::NotEnoughResourcesInBank);
                }
                let total_player_tokens = player_resources.iter().sum::<i8>();
                if total_player_tokens - self.amount > MAX_INVENTORY_TOKENS {
                    return Err(BankTransactionError::MaxTokensPerPlayerExceeded);
                }
            }
            1.. => {
                if player_resources[self.resource] - self.amount < 0 {
                    return Err(BankTransactionError::NotEnoughResourcesInPlayer);
                }
            }
        };

        Ok(())
    }

    pub fn transact<ActorType, T>(&self, game: &mut T) -> Result<BankTransactionSuccess, BankTransactionError>
        where ActorType: KnowableActorData,
              T: KnowableGameData<ActorType>
    {
        self.can_transact(game)?;

        game.get_actor_at_index_mut(self.player)
            .ok_or(BankTransactionError::PlayerDoesNotExist)?
            .owned_resources_mut()[self.resource] -= self.amount;

        game.bank_resources_mut()[self.resource] += self.amount;

        Ok(BankTransactionSuccess::FullTransaction)
    }
}

pub fn get_transaction_sequence_tokens(player: PlayerSelection, amount: i8, resources: &[ResourceTokenType]) -> Vec<BankTransaction> {
    resources.iter()
        .map(|resource| BankTransaction{
            player,
            resource: *resource,
            amount
        })
        .collect()
}

pub fn get_transaction_sequence(player: PlayerSelection, amount: i8, resources: &[ResourceType]) -> Vec<BankTransaction> {
    let tokens = resources.iter()
        .map(|resource| CostType(*resource))
        .collect::<Vec<ResourceTokenType>>();
    get_transaction_sequence_tokens(player, amount, &tokens)
}

#[derive(Debug, PartialEq)]

pub struct BankTransaction{
    pub player: PlayerSelection,
    pub resource: ResourceTokenType,
    /// Positive for deposit into bank, negative for withdrawal from bank
    pub amount: i8
}

#[derive(Debug, PartialEq)]
pub enum BankTransactionError{
    NotEnoughResourcesInBank,
    NotEnoughResourcesInPlayer,
    MaxTokensPerPlayerExceeded,
    PlayerDoesNotExist
}

#[derive(Debug, PartialEq)]
pub enum BankTransactionSuccess {
    FullTransaction
}


#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::constants::PlayerSelection::*;
    use crate::constants::ResourceType::Diamond;
    use super::*;
    use ResourceTokenType::*;

    use super::BankTransactionError::*;
    use super::BankTransactionSuccess::*;
    #[test]
    fn when_deposit_single_token__deposits() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 5;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Diamond] = 1;

        let transaction = BankTransaction{
            player: PlayerSelection1,
            resource: CostType(Diamond),
            amount: 1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Ok(FullTransaction));
        assert_eq!(game.bank_resources[Diamond], 6);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Diamond], 0);
    }

    #[test]
    fn when_deposit_missing_token__fails_not_enough_player() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 5;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Diamond] = 0;

        let transaction = BankTransaction{
            player: PlayerSelection1,
            resource: CostType(Diamond),
            amount: 1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Err(NotEnoughResourcesInPlayer));
        assert_eq!(game.bank_resources[Diamond], 5);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Diamond], 0);
    }
    #[test]
    fn when_player_missing__fails() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 5;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Diamond] = 0;

        let transaction = BankTransaction{
            player: PlayerSelection3,
            resource: CostType(Diamond),
            amount: 1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Err(PlayerDoesNotExist));
        assert_eq!(game.bank_resources[Diamond], 5);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Diamond], 0);
    }

    #[test]
    fn when_withdraw_single_token__withdraws() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 5;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Diamond] = 1;

        let transaction = BankTransaction{
            player: PlayerSelection1,
            resource: CostType(Diamond),
            amount: -1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Ok(FullTransaction));
        assert_eq!(game.bank_resources[Diamond], 4);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Diamond], 2);
    }
    #[test]
    fn when_withdraw_empty_bank__fail_not_enough_bank() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Gold] = 0;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Gold] = 1;

        let transaction = BankTransaction{
            player: PlayerSelection1,
            resource: CostType(Diamond),
            amount: -1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Err(NotEnoughResourcesInBank));
        assert_eq!(game.bank_resources[Gold], 0);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Gold], 1);
    }

    #[test]
    fn when_withdraw_full_player__fail_max_tokens_exceeded() {
        let mut game = crate::game_actions::test_utils::get_test_game(2);
        game.bank_resources[Diamond] = 5;
        game.actors[PlayerSelection1].as_mut().unwrap().resource_tokens[Gold] = MAX_INVENTORY_TOKENS;

        let transaction = BankTransaction{
            player: PlayerSelection1,
            resource: CostType(Diamond),
            amount: -1
        };

        let result = transaction.transact(&mut game);

        assert_eq!(result, Err(MaxTokensPerPlayerExceeded));
        assert_eq!(game.bank_resources[Diamond], 5);
        assert_eq!(game.get_actor_at_index(PlayerSelection1).unwrap().resource_tokens[Diamond], 0);
    }
    
}

