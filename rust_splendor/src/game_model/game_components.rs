use crate::game_model::constants::{COST_TYPE_COUNT};

#[derive(Debug)]
pub struct Card {
    pub id: u32,
    pub cost: [u8; COST_TYPE_COUNT],
    pub returns: [u8; COST_TYPE_COUNT],
    pub points: u8,
}

#[derive(Debug)]
pub struct Noble {
    pub id: u32,
    pub cost: [u8; COST_TYPE_COUNT],
    pub points: u8,
}

impl Noble {
    pub fn new() -> Noble {
        Noble {
            id: 257,
            cost: [104; COST_TYPE_COUNT],
            points: 105,
        }
    }
}

impl Card {
    pub fn new() -> Card {
        Card {
            id: 256,
            cost: [106; COST_TYPE_COUNT],
            returns: [107; COST_TYPE_COUNT],
            points: 108,
        }
    }
}