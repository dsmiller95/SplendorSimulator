use crate::game_model::constants::{CARD_TIER_COUNT, COST_TYPE_COUNT, MAX_NOBLES, MAX_RESERVED_CARDS, RESOURCE_TYPE_COUNT};
use super::constants::MAX_PLAYER_COUNT;

/// Data in this struct represents all knowable information about the game, which can be passed
/// to and from the AI.
/// this structure is designed to be rapidly blitted to the ai, so it should be small and ideally
///     match the structure of the data we will send to the ai.
#[derive(Debug)]
pub struct GameKnowable {
    pub actors: [Option<ActorKnowable>; MAX_PLAYER_COUNT],
    pub available_nobles: [Option<NobleKnowable>; MAX_NOBLES],
    pub bank_resources: [u8; RESOURCE_TYPE_COUNT],
    pub card_rows: [CardRowKnowable; CARD_TIER_COUNT],
}

#[derive(Debug)]
pub struct ActorKnowable{
    pub resource_tokens : [u8; RESOURCE_TYPE_COUNT],
    pub resources_from_cards : [u8; COST_TYPE_COUNT],
    pub current_points : u8,
    pub reserved_cards: [Option<CardKnowable>; MAX_RESERVED_CARDS],
}

#[derive(Debug)]
pub struct CardRowKnowable {
    pub open_cards: [Option<CardKnowable>; CARD_TIER_COUNT],
    pub hidden_card: Option<CardKnowable>,
}

#[derive(Debug)]
pub struct CardKnowable {
    pub cost: [u8; COST_TYPE_COUNT],
    pub returns: [u8; COST_TYPE_COUNT],
    pub points: u8,
}

#[derive(Debug)]
pub struct NobleKnowable {
    pub cost: [u8; COST_TYPE_COUNT],
    pub points: u8,
}

impl GameKnowable {
    pub fn new() -> GameKnowable {
        GameKnowable {
            actors: std::array::from_fn(|_| Some(ActorKnowable::new())),
            available_nobles: std::array::from_fn(|_| Some(NobleKnowable::new())),
            bank_resources: [100; RESOURCE_TYPE_COUNT],
            card_rows: std::array::from_fn(|_| CardRowKnowable::new()),
        }
    }
}

impl ActorKnowable {
    pub fn new() -> ActorKnowable {
        ActorKnowable {
            resource_tokens: [101; RESOURCE_TYPE_COUNT],
            resources_from_cards: [102; COST_TYPE_COUNT],
            current_points: 103,
            reserved_cards: std::array::from_fn(|_| None),
        }
    }
}

impl CardRowKnowable {
    pub fn new() -> CardRowKnowable {
        CardRowKnowable {
            open_cards: std::array::from_fn(|_| Some(CardKnowable::new())),
            hidden_card: None,
        }
    }
}

impl NobleKnowable {
    pub fn new() -> NobleKnowable {
        NobleKnowable {
            cost: [104; COST_TYPE_COUNT],
            points: 105,
        }
    }
}

impl CardKnowable {
    pub fn new() -> CardKnowable {
        CardKnowable {
            cost: [106; COST_TYPE_COUNT],
            returns: [107; COST_TYPE_COUNT],
            points: 108,
        }
    }
}