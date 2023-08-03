mod card_picks {
    use std::ops::{Index, IndexMut};
    use crate::game_model::constants::{CARD_COUNT_PER_TIER, CARD_TIER_COUNT, CardPickInTier, CardTier, OpenCardPickInTier};
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

    impl Index<CardTier> for [CardRowSized; CARD_TIER_COUNT] {
        type Output = CardRowSized;
        fn index(&self, index: CardTier) -> &Self::Output {
            &self[index as usize]
        }
    }
    impl IndexMut<CardTier> for [CardRowSized; CARD_TIER_COUNT] {
        fn index_mut(&mut self, index: CardTier) -> &mut Self::Output {
            &mut self[index as usize]
        }
    }

    impl Index<OpenCardPickInTier> for [Option<Card>; CARD_COUNT_PER_TIER] {
        type Output = Option<Card>;
        fn index(&self, index: OpenCardPickInTier) -> &Self::Output {
            &self[index as usize]
        }
    }
    impl IndexMut<OpenCardPickInTier> for [Option<Card>; CARD_COUNT_PER_TIER] {
        fn index_mut(&mut self, index: OpenCardPickInTier) -> &mut Self::Output {
            &mut self[index as usize]
        }
    }
}

mod resource_tokens{
    use std::ops::{Index, IndexMut};
    use std::slice::Iter;
    use crate::game_model::constants::{RESOURCE_TOKEN_COUNT, RESOURCE_TYPE_COUNT, ResourceTokenType, ResourceType};

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
