pub const MAX_NOBLES: usize = 5;
pub const MAX_INVENTORY_TOKENS : i8 = 10;

use std::ops::{Index, IndexMut};
use seq_macro::seq;

macro_rules! sequential_enum {
    ($const_name:ident, $enum_name:ident, $count:expr) => {
        pub const $const_name: usize = $count;
        seq!(N in 1..=$count {
            #[derive(Debug, Copy, Clone)]
            pub enum $enum_name {
                #(
                    $enum_name~N,
                )*
            }
        });
    }
}


macro_rules! indexable_sequential_enum {
    ($const_name:ident, $enum_name:ident, $count: expr) => {
        sequential_enum!($const_name, $enum_name, $count);
        seq!(N in 1..=$count {
            impl<T> Index<$enum_name> for [T; $const_name] {
                type Output = T;
                fn index(&self, index: $enum_name) -> &Self::Output {
                    match index {
                        #($enum_name::$enum_name~N => &self[N - 1],)*
                    }
                }
            }
            impl<T> IndexMut<$enum_name> for [T; $const_name] {
                fn index_mut(&mut self, index: $enum_name) -> &mut Self::Output {
                    match index {
                        #($enum_name::$enum_name~N => &mut self[N - 1],)*
                    }
                }
            }
        });
    }
}
indexable_sequential_enum!(CARD_COUNT_PER_TIER, OpenCardPickInTier, 4);

#[derive(Debug, Copy, Clone)]
pub enum CardPickInTier {
    HiddenCard,
    OpenCard(OpenCardPickInTier)
}


indexable_sequential_enum!(CARD_TIER_COUNT, CardTier, 3);

#[derive(Debug, Copy, Clone)]
pub struct CardPickOnBoard {
    pub tier: CardTier,
    pub pick_in_tier: CardPickInTier,
}

indexable_sequential_enum!(MAX_PLAYER_COUNT, PlayerSelection, 4);
indexable_sequential_enum!(MAX_RESERVED_CARDS, ReservedCardSelection, 3);

#[derive(Debug, Copy, Clone)]
pub struct CardPickInReservedCards {
    pub player_index : PlayerSelection,
    pub reserved_card_index : ReservedCardSelection,
}

#[derive(Debug, Copy, Clone)]
pub enum GlobalCardPick {
    OnBoard(CardPickOnBoard),
    Reserved(CardPickInReservedCards),
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