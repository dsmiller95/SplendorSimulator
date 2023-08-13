use crate::constants::{MAX_RESERVED_CARDS, RESOURCE_TOKEN_COUNT, ResourceType};
use crate::constants::ResourceTokenType::Gold;
use crate::game_actions::knowable_game_data::KnowableActorData;
use crate::game_model::game_components::Card;

use crate::game_model::game_sized::{ActorSized};

impl KnowableActorData for ActorSized {
    fn owned_resources(&self) -> &[i8; RESOURCE_TOKEN_COUNT] {
        &self.resource_tokens
    }

    fn owned_resources_mut(&mut self) -> &mut [i8; RESOURCE_TOKEN_COUNT] {
        &mut self.resource_tokens
    }

    fn can_afford_card(&self, card: &Card) -> bool {
        let mut total_deficit = 0;
        for &resource in ResourceType::iterator() {
            let deficit = card.cost[resource] 
                - self.resources_from_cards[resource]
                - self.resource_tokens[resource];
            if deficit > 0 {
                total_deficit += deficit;
            }
        }
        
        let gold_tokens = self.resource_tokens[Gold];
        
        
        total_deficit > 0 && gold_tokens >= total_deficit
    }

    fn reserved_cards(&self) -> &[Option<Card>; MAX_RESERVED_CARDS] {
        &self.reserved_cards
    }
}