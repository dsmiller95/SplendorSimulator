use crate::game_model::constants::{CARD_COUNT_PER_TIER, CARD_TIER_COUNT, COST_TYPE_COUNT, MAX_NOBLES, MAX_RESERVED_CARDS, RESOURCE_TYPE_COUNT};
use crate::game_model::game_components::{Card, Noble};
use super::constants::MAX_PLAYER_COUNT;

/// Data in this struct represents all relevant information about the game, which can be passed
/// to and from the AI.
/// This structure must be fully sized, and not contain any references to other data.
/// this structure is designed to be rapidly blitted to the ai, so it should be small and ideally
///     match the structure of the data we will send to the ai.
#[derive(Debug)]
pub struct GameSized {
    /// assume the 1st actor in this list is the player whose turn it is.
    pub actors: [Option<ActorSized>; MAX_PLAYER_COUNT],
    pub available_nobles: [Option<Noble>; MAX_NOBLES],
    pub bank_resources: [u8; RESOURCE_TYPE_COUNT],
    pub card_rows: [CardRowSized; CARD_TIER_COUNT],
}

#[derive(Debug)]
pub struct ActorSized {
    pub resource_tokens : [u8; RESOURCE_TYPE_COUNT],
    pub resources_from_cards : [u8; COST_TYPE_COUNT],
    pub current_points : u8,
    pub reserved_cards: [Option<Card>; MAX_RESERVED_CARDS],
}

#[derive(Debug)]
pub struct CardRowSized {
    pub open_cards: [Option<Card>; CARD_COUNT_PER_TIER],
    /// the top of the pile. typically invisible to the players, but included here because
    /// it can be involved in executing a specific action.
    pub hidden_card: Option<Card>,
}

impl GameSized {
    pub fn new() -> GameSized {
        GameSized {
            actors: std::array::from_fn(|_| Some(ActorSized::new())),
            available_nobles: std::array::from_fn(|_| Some(Noble::new())),
            bank_resources: [100; RESOURCE_TYPE_COUNT],
            card_rows: std::array::from_fn(|_| CardRowSized::new()),
        }
    }
}

impl ActorSized {
    pub fn new() -> ActorSized {
        ActorSized {
            resource_tokens: [101; RESOURCE_TYPE_COUNT],
            resources_from_cards: [102; COST_TYPE_COUNT],
            current_points: 103,
            reserved_cards: std::array::from_fn(|_| None),
        }
    }
}

impl CardRowSized {
    pub fn new() -> CardRowSized {
        CardRowSized {
            open_cards: std::array::from_fn(|_| Some(Card::new())),
            hidden_card: None,
        }
    }
}
