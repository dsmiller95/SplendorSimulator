pub const MAX_PLAYER_COUNT: usize = 4;
pub const MAX_NOBLES: usize = 5;
pub const MAX_RESERVED_CARDS : usize = 3;
pub const MAX_INVENTORY_TOKENS : i8 = 10;

pub const CARD_TIER_COUNT: usize = 3;
pub const CARD_COUNT_PER_TIER: usize = 4;

#[derive(Debug, Copy, Clone)]
pub enum CardTier {
    Tier1 = 0,
    Tier2 = 1,
    Tier3 = 2,
}

#[derive(Debug, Copy, Clone)]
pub enum CardPickInTier {
    HiddenCard,
    OpenCard(OpenCardPickInTier)
}

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum OpenCardPickInTier {
    OpenCard1 = 0,
    OpenCard2 = 1,
    OpenCard3 = 2,
    OpenCard4 = 3,
}

#[derive(Debug, Copy, Clone)]
pub struct CardPick {
    pub tier: CardTier,
    pub pick_in_tier: CardPickInTier,
}


pub const RESOURCE_TOKEN_COUNT: usize = 6;
pub const RESOURCE_TYPE_COUNT: usize = RESOURCE_TOKEN_COUNT - 1;
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum ResourceType{
    RUBY = 0,
    EMERALD = 1,
    SAPPHIRE = 2,
    DIAMOND = 3,
    ONYX = 4,
}

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum ResourceTokenType {
    CostType(ResourceType),
    GOLD = 5,
}