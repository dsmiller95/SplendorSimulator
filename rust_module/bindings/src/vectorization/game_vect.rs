use crate::vectorization::to_vect::*;
use rust_splendor::{game_model::{game_full::GameModel, actor::Actor, card::CardRow, game_components::*}, constants::*};

impl ToVect for GameModel {
    fn vect_size() -> usize {
        Option::<Actor>::vect_size() * MAX_PLAYER_COUNT
        + Option::<Noble>::vect_size() * MAX_NOBLES
        + RESOURCE_TOKEN_COUNT
        + CardRow::vect_size() * CARD_TIER_COUNT
    }
    fn describe_slice() -> Vec<ToVectDescription>{
        let mut result = vec![];
        
        let actor_description = Actor::describe_slice();
        result.extend(get_n_descriptions(MAX_PLAYER_COUNT, "player", &actor_description));

        let noble_description = Noble::describe_slice();
        result.extend(get_n_descriptions(MAX_NOBLES, "board_noble", &noble_description));

        
        let card_row_description = CardRow::describe_slice();
        result.extend(get_n_descriptions(CARD_TIER_COUNT, "tier", &card_row_description));

        result.extend(vec![ToVectDescription{
            name: "resources".into(),
            size: RESOURCE_TOKEN_COUNT,
        }]);

        result
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
    fn describe_slice() -> Vec<ToVectDescription>{
        let mut result = vec![];
        result.extend(vec![ToVectDescription{
            name: "temp_resources".into(),
            size: RESOURCE_TOKEN_COUNT,
        }]);
        result.extend(vec![ToVectDescription{
            name: "perm_resources".into(),
            size: RESOURCE_TYPE_COUNT,
        }]);
        result.extend(vec![ToVectDescription{
            name: "points".into(),
            size: 1,
        }]);
        result.extend(vec![ToVectDescription{
            name: "ordering".into(),
            size: 1,
        }]);
        let card_description = Card::describe_slice();
        result.extend(get_n_descriptions(MAX_RESERVED_CARDS, "reserved_card", &card_description));
        result
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
    fn describe_slice() -> Vec<ToVectDescription>{
        let mut result = vec![];
        result.extend(vec![ToVectDescription{
            name: "costs".into(),
            size: RESOURCE_TYPE_COUNT,
        }]);
        result.extend(vec![ToVectDescription{
            name: "returns".into(),
            size: RESOURCE_TYPE_COUNT,
        }]);
        result.extend(vec![ToVectDescription{
            name: "points".into(),
            size: 1,
        }]);
        result
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!();
    }
}

impl ToVect for Noble{
    fn vect_size() -> usize {
        RESOURCE_TYPE_COUNT
        + 1
    }
    fn describe_slice() -> Vec<ToVectDescription>{
        let mut result = vec![];
        result.extend(vec![ToVectDescription{
            name: "costs".into(),
            size: RESOURCE_TYPE_COUNT,
        }]);
        result.extend(vec![ToVectDescription{
            name: "points".into(),
            size: 1,
        }]);
        result
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!();
    }
}

impl ToVect for CardRow{
    fn vect_size() -> usize {
        Option::<Card>::vect_size()
        + Option::<Card>::vect_size() * CARD_COUNT_PER_TIER
    }
    fn describe_slice() -> Vec<ToVectDescription>{
        let mut result = vec![];
        let card_description = Card::describe_slice();
        result.extend(get_n_descriptions(CARD_COUNT_PER_TIER, "open_card", &card_description));
        result.extend(card_description.iter().map(|desc| ToVectDescription{
            name: format!("hidden_card_{}", desc.name).into(),
            size: desc.size,
        }));
        result
    }
    fn populate_slice(&self, slice: &mut [f32]) {
        todo!()
    }
}