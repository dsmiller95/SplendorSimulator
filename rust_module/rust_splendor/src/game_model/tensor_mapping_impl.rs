use crate::constants::*;
use crate::game_model::actor::Actor;
use crate::game_model::card::CardRow;
use crate::game_model::game_components::{Card, Noble};
use crate::game_model::game_full::GameModel;
use crate::game_model::tensor_mapping::*;

impl MapsToTensorAbilities for GameModel {
    fn build_mapping_description(mut builder: TensorDescriptionBuilder) -> TensorDescriptionBuilder {

        for actor_n in 0..MAX_PLAYER_COUNT {
            Actor::build_mapping_description(builder.with_scope(&format!("player_{}", actor_n)));
        }
        for noble_n in 0..MAX_NOBLES {
            Noble::build_mapping_description(builder.with_scope(&format!("board_noble_{}", noble_n)));
        }
        for tier_n in 0..CARD_TIER_COUNT {
            CardRow::build_mapping_description(builder.with_scope(&format!("tier_{}", tier_n)));
        }

        builder.append_mapping("resources", RESOURCE_TOKEN_COUNT);
        builder
    }
}

impl MapsToTensorAbilities for Actor {
    fn build_mapping_description(mut builder: TensorDescriptionBuilder) -> TensorDescriptionBuilder {
        builder.append_mapping("temp_resources", RESOURCE_TOKEN_COUNT);
        builder.append_mapping("perm_resources", RESOURCE_TYPE_COUNT);
        builder.append_mapping("points", 1);
        for reserved_n in 0..MAX_RESERVED_CARDS {
            Card::build_mapping_description(builder.with_scope(&format!("reserved_card_{}", reserved_n)));
        }
        builder
    }
}
impl MapsToTensorAbilities for Noble {
    fn build_mapping_description(mut builder: TensorDescriptionBuilder) -> TensorDescriptionBuilder {
        builder.append_mapping("costs", RESOURCE_TYPE_COUNT);
        builder.append_mapping("points", 1);
        builder
    }
}

impl MapsToTensorAbilities for CardRow {
    fn build_mapping_description(mut builder: TensorDescriptionBuilder) -> TensorDescriptionBuilder {
        for card_n in 0..CARD_COUNT_PER_TIER {
            Card::build_mapping_description(builder.with_scope(&format!("open_card_{}", card_n)));
        }
        Card::build_mapping_description(builder.with_scope("hidden_card"));
        builder
    }
}

impl MapsToTensorAbilities for Card {
    fn build_mapping_description(mut builder: TensorDescriptionBuilder) -> TensorDescriptionBuilder {
        builder.append_mapping("costs", RESOURCE_TYPE_COUNT);
        builder.append_mapping("returns", RESOURCE_TYPE_COUNT);
        builder.append_mapping("points", 1);
        builder
    }
}

#[cfg(test)]
mod test {
    use crate::game_model::game_full::GameModel;
    use crate::game_model::tensor_mapping::*;

    #[test]
    fn test_mapping_size() {
        let mut tensor_description = TensorDescription {
            tensor_mapping: Vec::new(),
        };
        GameModel::build_mapping_description(TensorDescriptionBuilder::new(&mut tensor_description));

        assert_eq!(tensor_description.tensor_mapping.iter().map(|x| x.range.len()).sum::<usize>(), 381);
    }

    #[test]
    fn test_mapping_names() {
        let expected_descriptions = [
            "player_0_temp_resources",
            "player_0_perm_resources",
            "player_0_points",
            "player_0_reserved_card_0_costs",
            "player_0_reserved_card_0_returns",
            "player_0_reserved_card_0_points",
            "player_0_reserved_card_1_costs",
            "player_0_reserved_card_1_returns",
            "player_0_reserved_card_1_points",
            "player_0_reserved_card_2_costs",
            "player_0_reserved_card_2_returns",
            "player_0_reserved_card_2_points",
            "player_1_temp_resources",
            "player_1_perm_resources",
            "player_1_points",
            "player_1_reserved_card_0_costs",
            "player_1_reserved_card_0_returns",
            "player_1_reserved_card_0_points",
            "player_1_reserved_card_1_costs",
            "player_1_reserved_card_1_returns",
            "player_1_reserved_card_1_points",
            "player_1_reserved_card_2_costs",
            "player_1_reserved_card_2_returns",
            "player_1_reserved_card_2_points",
            "player_2_temp_resources",
            "player_2_perm_resources",
            "player_2_points",
            "player_2_reserved_card_0_costs",
            "player_2_reserved_card_0_returns",
            "player_2_reserved_card_0_points",
            "player_2_reserved_card_1_costs",
            "player_2_reserved_card_1_returns",
            "player_2_reserved_card_1_points",
            "player_2_reserved_card_2_costs",
            "player_2_reserved_card_2_returns",
            "player_2_reserved_card_2_points",
            "player_3_temp_resources",
            "player_3_perm_resources",
            "player_3_points",
            "player_3_reserved_card_0_costs",
            "player_3_reserved_card_0_returns",
            "player_3_reserved_card_0_points",
            "player_3_reserved_card_1_costs",
            "player_3_reserved_card_1_returns",
            "player_3_reserved_card_1_points",
            "player_3_reserved_card_2_costs",
            "player_3_reserved_card_2_returns",
            "player_3_reserved_card_2_points",
            "board_noble_0_costs",
            "board_noble_0_points",
            "board_noble_1_costs",
            "board_noble_1_points",
            "board_noble_2_costs",
            "board_noble_2_points",
            "board_noble_3_costs",
            "board_noble_3_points",
            "board_noble_4_costs",
            "board_noble_4_points",
            "tier_0_open_card_0_costs",
            "tier_0_open_card_0_returns",
            "tier_0_open_card_0_points",
            "tier_0_open_card_1_costs",
            "tier_0_open_card_1_returns",
            "tier_0_open_card_1_points",
            "tier_0_open_card_2_costs",
            "tier_0_open_card_2_returns",
            "tier_0_open_card_2_points",
            "tier_0_open_card_3_costs",
            "tier_0_open_card_3_returns",
            "tier_0_open_card_3_points",
            "tier_0_hidden_card_costs",
            "tier_0_hidden_card_returns",
            "tier_0_hidden_card_points",
            "tier_1_open_card_0_costs",
            "tier_1_open_card_0_returns",
            "tier_1_open_card_0_points",
            "tier_1_open_card_1_costs",
            "tier_1_open_card_1_returns",
            "tier_1_open_card_1_points",
            "tier_1_open_card_2_costs",
            "tier_1_open_card_2_returns",
            "tier_1_open_card_2_points",
            "tier_1_open_card_3_costs",
            "tier_1_open_card_3_returns",
            "tier_1_open_card_3_points",
            "tier_1_hidden_card_costs",
            "tier_1_hidden_card_returns",
            "tier_1_hidden_card_points",
            "tier_2_open_card_0_costs",
            "tier_2_open_card_0_returns",
            "tier_2_open_card_0_points",
            "tier_2_open_card_1_costs",
            "tier_2_open_card_1_returns",
            "tier_2_open_card_1_points",
            "tier_2_open_card_2_costs",
            "tier_2_open_card_2_returns",
            "tier_2_open_card_2_points",
            "tier_2_open_card_3_costs",
            "tier_2_open_card_3_returns",
            "tier_2_open_card_3_points",
            "tier_2_hidden_card_costs",
            "tier_2_hidden_card_returns",
            "tier_2_hidden_card_points",
            "resources"
        ];

        let mut tensor_description = TensorDescription {
            tensor_mapping: Vec::new(),
        };
        GameModel::build_mapping_description(TensorDescriptionBuilder::new(&mut tensor_description));


        let actual_descriptions = tensor_description.tensor_mapping.iter().map(|x| x.name.as_str()).collect::<Vec<&str>>();
        assert_eq!(actual_descriptions, expected_descriptions);

    }
}