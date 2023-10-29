pub const MAX_NOBLES: usize = 5;
pub const MAX_INVENTORY_TOKENS : i8 = 10;

use std::ops::{Index, IndexMut};
use seq_macro::seq;
use std::slice::Iter;
macro_rules! sequential_enum {
    ($const_name:ident, $enum_name:ident, $count:expr) => {
        pub const $const_name: usize = $count;
        seq!(N in 1..=$count {
            #[derive(Debug, PartialEq, Copy, Clone)]
            pub enum $enum_name {
                #(
                    $enum_name~N,
                )*
            }
            impl $enum_name {
                pub fn iterator() -> Iter<'static, $enum_name> {
                    static VALS: [$enum_name; $const_name] = [
                        #(
                            $enum_name::$enum_name~N,
                        )*
                    ];
                    VALS.iter()
                }
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


#[derive(Debug, PartialEq, Copy, Clone)]
pub enum CardPickInTier {
    HiddenCard,
    OpenCard(OpenCardPickInTier)
}


indexable_sequential_enum!(CARD_TIER_COUNT, CardTier, 3);

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CardPickOnBoard {
    pub tier: CardTier,
    pub pick: CardPickInTier,
}

indexable_sequential_enum!(MAX_PLAYER_COUNT, PlayerSelection, 4);
indexable_sequential_enum!(MAX_RESERVED_CARDS, ReservedCardSelection, 3);

#[derive(Debug, Copy, Clone)]
pub struct CardPickInReservedCards {
    pub player_index : PlayerSelection,
    pub reserved_card: ReservedCardSelection,
}

#[derive(Debug, Copy, Clone)]
pub enum GlobalCardPick {
    OnBoard(CardPickOnBoard),
    Reserved(CardPickInReservedCards),
}
#[derive(Debug, Copy, Clone)]
pub enum PlayerCardPick{
    OnBoard(CardPickOnBoard),
    Reserved(ReservedCardSelection),
}

impl From<CardPickOnBoard> for GlobalCardPick {
    fn from(card_pick: CardPickOnBoard) -> Self {
        GlobalCardPick::OnBoard(card_pick)
    }
}
impl From<CardPickInReservedCards> for GlobalCardPick {
    fn from(card_pick: CardPickInReservedCards) -> Self {
        GlobalCardPick::Reserved(card_pick)
    }
}

impl From<CardPickOnBoard> for PlayerCardPick {
    fn from(card_pick: CardPickOnBoard) -> Self {
        PlayerCardPick::OnBoard(card_pick)
    }
}

impl PlayerCardPick {
    pub fn as_global(&self, player_index: PlayerSelection) -> GlobalCardPick {
        match self {
            PlayerCardPick::OnBoard(card_pick) => (*card_pick).into(),
            PlayerCardPick::Reserved(reserved_card) => CardPickInReservedCards{
                player_index,
                reserved_card: *reserved_card,
            }.into(),
        }
    }
}

pub fn reserved_card(player_index: PlayerSelection, reserved_card: ReservedCardSelection) -> GlobalCardPick {
    GlobalCardPick::Reserved(CardPickInReservedCards {
        player_index,
        reserved_card,
    })
}


pub const RESOURCE_TOKEN_COUNT: usize = 6;
pub type ResourceTokenBank = [i8; RESOURCE_TOKEN_COUNT];
pub const RESOURCE_TYPE_COUNT: usize = RESOURCE_TOKEN_COUNT - 1;
pub type ResourceAmountFlags = [i8; RESOURCE_TYPE_COUNT];
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum ResourceType{
    Ruby = 0,
    Emerald = 1,
    Sapphire = 2,
    Diamond = 3,
    Onyx = 4,
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(u8)]
pub enum ResourceTokenType {
    CostType(ResourceType),
    Gold = 5,
}