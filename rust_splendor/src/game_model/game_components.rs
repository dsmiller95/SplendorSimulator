use crate::game_model::constants::{RESOURCE_TYPE_COUNT};

#[derive(Debug)]
pub struct Card {
    pub id: u32,
    pub cost: [i8; RESOURCE_TYPE_COUNT],
    pub returns: [i8; RESOURCE_TYPE_COUNT],
    pub points: i8,
}

#[derive(Debug)]
pub struct Noble {
    pub id: u32,
    pub cost: [i8; RESOURCE_TYPE_COUNT],
    pub points: i8,
}

impl Noble {
    pub fn new() -> Noble {
        Noble {
            id: 257,
            cost: [104; RESOURCE_TYPE_COUNT],
            points: 105,
        }
    }
}

impl Card {
    pub fn new() -> Card {
        Card {
            id: 256,
            cost: [106; RESOURCE_TYPE_COUNT],
            returns: [107; RESOURCE_TYPE_COUNT],
            points: 108,
        }
    }
}