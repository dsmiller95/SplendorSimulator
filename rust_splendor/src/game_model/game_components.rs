use crate::constants::{RESOURCE_TYPE_COUNT, ResourceAmountFlags};

#[derive(Debug)]
pub struct Card {
    pub id: u32,
    pub cost: ResourceAmountFlags,
    pub returns: ResourceAmountFlags,
    pub points: i8,
}

#[derive(Debug)]
pub struct Noble {
    pub id: u32,
    pub cost: ResourceAmountFlags,
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
            cost: [0, 0, 0, 0, 0],
            returns: [107; RESOURCE_TYPE_COUNT],
            points: 108,
        }
    }
    pub fn with_id(mut self, id: u32) -> Self {
        self.id = id;
        self
    }
    pub fn with_cost(mut self, cost: ResourceAmountFlags) -> Self {
        self.cost = cost;
        self
    }
}