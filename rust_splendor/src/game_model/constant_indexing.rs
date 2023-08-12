mod card_picks {
    use std::ops::{Index, IndexMut};
    use crate::constants::{CardPickInTier};
    use crate::game_model::game_components::Card;
    use crate::game_model::game_sized::CardRowSized;

    impl Index<CardPickInTier> for CardRowSized {
        type Output = Option<Card>;
        fn index(&self, index: CardPickInTier) -> &Self::Output {
            match index {
                CardPickInTier::HiddenCard => &self.hidden_card,
                CardPickInTier::OpenCard(x) => &self.open_cards[x],
            }
        }
    }
    impl IndexMut<CardPickInTier> for CardRowSized {
        fn index_mut(&mut self, index: CardPickInTier) -> &mut Self::Output {
            match index {
                CardPickInTier::HiddenCard => &mut self.hidden_card,
                CardPickInTier::OpenCard(x) => &mut self.open_cards[x],
            }
        }
    }
}

mod resource_tokens{
    use std::ops::{Index, IndexMut};
    use std::slice::Iter;
    use crate::constants::{RESOURCE_TOKEN_COUNT, RESOURCE_TYPE_COUNT, ResourceTokenType, ResourceType};

    impl ResourceType {
        pub fn iterator() -> Iter<'static, ResourceType> {
            static DIRECTIONS: [ResourceType; 5] = [
                ResourceType::RUBY,
                ResourceType::EMERALD,
                ResourceType::SAPPHIRE,
                ResourceType::DIAMOND,
                ResourceType::ONYX];
            DIRECTIONS.iter()
        }
    }


    impl Index<ResourceType> for [i8; RESOURCE_TYPE_COUNT] {
        type Output = i8;
        fn index(&self, index: ResourceType) -> &Self::Output {
            &self[index as usize]
        }
    }
    impl IndexMut<ResourceType> for [i8; RESOURCE_TYPE_COUNT] {
        fn index_mut(&mut self, index: ResourceType) -> &mut Self::Output {
            &mut self[index as usize]
        }
    }
    
    impl Index<ResourceType> for [i8; RESOURCE_TOKEN_COUNT] {
        type Output = i8;
        fn index(&self, index: ResourceType) -> &Self::Output {
            &self[index as usize]
        }
    }
    impl IndexMut<ResourceType> for [i8; RESOURCE_TOKEN_COUNT] {
        fn index_mut(&mut self, index: ResourceType) -> &mut Self::Output {
            &mut self[index as usize]
        }
    }
    
    impl Index<ResourceTokenType> for [i8; RESOURCE_TOKEN_COUNT] {
        type Output = i8;
        fn index(&self, index: ResourceTokenType) -> &Self::Output {
            match index {
                ResourceTokenType::CostType(cost) => &self[cost],
                ResourceTokenType::GOLD => &self[5],
            }
        }
    }
    impl IndexMut<ResourceTokenType> for [i8; RESOURCE_TOKEN_COUNT] {
        fn index_mut(&mut self, index: ResourceTokenType) -> &mut Self::Output {
            match index {
                ResourceTokenType::CostType(cost) => &mut self[cost],
                ResourceTokenType::GOLD => &mut self[5],
            }
        }
    }
    
}