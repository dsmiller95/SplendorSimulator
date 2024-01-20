use crate::vectorization::ToVect;
use rust_splendor::{game_model::{game_full::GameModel, actor::Actor, card::CardRow, game_components::*}, constants::*};

impl ToVect for GameModel {
    fn vect_size() -> usize {
        Option::<Actor>::vect_size() * MAX_PLAYER_COUNT
        + Option::<Noble>::vect_size() * MAX_NOBLES
        + RESOURCE_TOKEN_COUNT
        + CardRow::vect_size() * CARD_TIER_COUNT
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}

impl ToVect for Actor{
    fn vect_size() -> usize {
        RESOURCE_TOKEN_COUNT
        + RESOURCE_TYPE_COUNT
        + 1
        + 1
        + Option::<Card>::vect_size() * MAX_RESERVED_CARDS
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}

impl ToVect for Card{
    fn vect_size() -> usize {
        RESOURCE_TYPE_COUNT
        + RESOURCE_TYPE_COUNT
        + 1
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}

impl ToVect for Noble{
    fn vect_size() -> usize {
        RESOURCE_TYPE_COUNT
        + 1
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}

impl ToVect for CardRow{
    fn vect_size() -> usize {
        Option::<Card>::vect_size()
        + Option::<Card>::vect_size() * CARD_COUNT_PER_TIER
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}