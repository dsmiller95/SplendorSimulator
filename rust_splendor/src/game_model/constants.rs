
pub const MAX_PLAYER_COUNT: usize = 4;
pub const MAX_NOBLES: usize = 5;
pub const MAX_RESERVED_CARDS : usize = 3;

pub const CARD_TIER_COUNT: usize = 3;
pub const CARD_COUNT_PER_TIER: usize = 4;

#[derive(Debug)]
pub enum CardTier {
    Tier1,
    Tier2,
    Tier3,
}


pub const RESOURCE_TYPE_COUNT: usize = 6;
pub const COST_TYPE_COUNT: usize = RESOURCE_TYPE_COUNT - 1;
#[derive(Debug)]
pub enum ResourceType{
    RUBY,
    EMERALD,
    SAPPHIRE,
    DIAMOND,
    ONYX,
    GOLD,
}

