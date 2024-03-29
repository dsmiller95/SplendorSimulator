use crate::constants::{MAX_RESERVED_CARDS, RESOURCE_TOKEN_COUNT, RESOURCE_TYPE_COUNT};
use crate::game_model::game_components::{Card, Noble};

#[derive(Debug)]
pub struct Actor {
    pub purchased_cards: Vec<Card>,
    pub claimed_nobles: Vec<Noble>,

    pub resource_tokens : [i8; RESOURCE_TOKEN_COUNT],
    pub resources_from_cards : [i8; RESOURCE_TYPE_COUNT],
    pub current_points : i8,
    pub reserved_cards: [Option<Card>; MAX_RESERVED_CARDS],
}
impl Actor {
    pub fn new() -> Actor {
        Actor {
            purchased_cards: Vec::new(),
            claimed_nobles: Vec::new(),
            resource_tokens: [0; RESOURCE_TOKEN_COUNT],
            resources_from_cards: [0; RESOURCE_TYPE_COUNT],
            current_points: 0,
            reserved_cards: std::array::from_fn(|_| None),
        }
    }
}
