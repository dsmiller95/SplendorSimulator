use crate::game_model::constants::ResourceType;
use crate::game_model::constants::ResourceTokenType::GOLD;
use crate::game_model::game_components::Card;

use crate::game_model::game_sized::{ActorSized};

impl ActorSized {
    pub fn can_afford_card(&self, card: &Card) -> bool {
        let mut total_deficit = 0;
        for &resource in ResourceType::iterator() {
            let deficit = card.cost[resource] 
                - self.resources_from_cards[resource]
                - self.resource_tokens[resource];
            if deficit > 0 {
                total_deficit += deficit;
            }
        }
        
        let gold_tokens = self.resource_tokens[GOLD];
        
        
        total_deficit > 0 && gold_tokens >= total_deficit
    }
}