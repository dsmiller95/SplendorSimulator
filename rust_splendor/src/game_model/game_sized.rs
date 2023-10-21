use crate::constants::{CARD_COUNT_PER_TIER, RESOURCE_TYPE_COUNT, MAX_RESERVED_CARDS, RESOURCE_TOKEN_COUNT};
use crate::game_model::game_components::Card;

#[derive(Debug)]
pub struct CardRowSized {
    pub open_cards: [Option<Card>; CARD_COUNT_PER_TIER],
    /// the top of the pile. typically invisible to the players, but included here because
    /// it can be involved in executing a specific action.
    pub hidden_card: Option<Card>,
}

#[derive(Debug)]
pub struct ActorSized {
    pub resource_tokens : [i8; RESOURCE_TOKEN_COUNT],
    pub resources_from_cards : [i8; RESOURCE_TYPE_COUNT],
    pub current_points : i8,
    pub reserved_cards: [Option<Card>; MAX_RESERVED_CARDS],
}

impl ActorSized {
    pub fn new() -> ActorSized {
        ActorSized {
            resource_tokens: [0; RESOURCE_TOKEN_COUNT],
            resources_from_cards: [0; RESOURCE_TYPE_COUNT],
            current_points: 0,
            reserved_cards: std::array::from_fn(|_| None),
        }
    }
}

impl CardRowSized {
    pub fn new() -> CardRowSized {
        CardRowSized {
            open_cards: std::array::from_fn(|_| None),
            hidden_card: None,
        }
    }
}
