use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Deserialize;
use rust_splendor;
use rust_splendor::game_model::game_components::{Card, Noble};
use rust_splendor::game_model::game_config::{GameConfig, TieredCard};

#[pyclass]
#[derive(Clone)]
pub struct SplendorConfig {
    pub wrapped_config: GameConfig,
}

#[pymethods]
impl SplendorConfig {
    #[classmethod]
    fn parse_config_csv(cls: &PyType, data: String) -> PyResult<SplendorConfig> {
        SplendorConfig::parse_config_csv_internal(data).map_err(|e| {
            PyValueError::new_err(e)
        }).map(|c| {
            SplendorConfig{
                wrapped_config: c,
            }
        })
    }
}

impl SplendorConfig{
    fn parse_config_csv_internal(data: String) -> Result<GameConfig, String> {
        let mut config = GameConfig::new();
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        for result in rdr.deserialize() {
            let record: SplendorConfigFormat = result.map_err(|e| {
                format!("Error parsing config CSV: {}", e)
            })?;

            let cost = [
                record.ruby_cost,
                record.emerald_cost,
                record.sapphire_cost,
                record.diamond_cost,
                record.onyx_cost,
            ];
            let returns = [
                record.ruby_return,
                record.emerald_return,
                record.sapphire_return,
                record.diamond_return,
                record.onyx_return,
            ];

            match record.card_type.as_str() {
                "Noble" => {
                    config.all_nobles.push(
                        Noble{
                            id: record.id,
                            points: record.points_return,
                            cost,
                        }
                    );
                },
                "Regular" => {
                    config.all_cards.push(TieredCard{
                        tier: record.card_level,
                        card: Card{
                            id: record.id,
                            cost,
                            returns,
                            points: record.points_return,
                        }
                    })
                },
                _ => {
                    return Err(
                        format!("Unknown card type: {}", record.card_type)
                    );
                }
            }
        }
        Ok(config)
    }
}


#[derive(Debug, Deserialize)]
struct SplendorConfigFormat {
    #[serde(rename = "Card ID")]        pub id: u32,
    #[serde(rename = "Card Type")]      pub card_type: String,
    #[serde(rename = "Card Level")]     pub card_level: u8,
    #[serde(rename = "Ruby Cost")]      pub ruby_cost: i8,
    #[serde(rename = "Emerald Cost")]   pub emerald_cost: i8,
    #[serde(rename = "Sapphire Cost")]  pub sapphire_cost: i8,
    #[serde(rename = "Diamond Cost")]   pub diamond_cost: i8,
    #[serde(rename = "Onyx Cost")]      pub onyx_cost: i8,
    #[serde(rename = "Ruby Return")]    pub ruby_return: i8,
    #[serde(rename = "Emerald Return")] pub emerald_return: i8,
    #[serde(rename = "Sapphire Return")]pub sapphire_return: i8,
    #[serde(rename = "Diamond Return")] pub diamond_return: i8,
    #[serde(rename = "Onyx Return")]    pub onyx_return: i8,
    #[serde(rename = "Points Return")]  pub points_return: i8,

}

#[cfg(test)]
mod test {
    #[test]
    fn test_parse_config_csv() {
        let csv = r"
Card ID,Card Type,Card Level,Ruby Cost,Emerald Cost,Sapphire Cost,Diamond Cost,Onyx Cost,Ruby Return,Emerald Return,Sapphire Return,Diamond Return,Onyx Return,Points Return
1,Noble,0,3,3,3,0,0,0,0,0,0,0,3
2,Noble,0,0,0,4,4,0,0,0,0,0,0,3
3,Noble,0,3,3,0,0,3,0,0,0,0,0,3
11,Regular,1,0,1,1,1,1,1,0,0,0,0,0
12,Regular,1,0,1,1,2,1,1,0,0,0,0,0
13,Regular,1,2,0,1,0,2,0,1,0,0,0,0
14,Regular,2,0,3,0,0,0,0,0,0,0,1,0
15,Regular,2,0,2,0,2,0,0,0,0,0,1,0
16,Regular,2,1,3,1,0,0,0,0,1,0,0,0
17,Regular,3,0,0,3,0,0,0,0,0,1,0,0";

        let config = super::SplendorConfig::parse_config_csv_internal(csv.to_string()).unwrap();
        assert_eq!(config.all_cards.len(), 7);
        assert_eq!(config.all_cards[0].card.id, 11);
        assert_eq!(config.all_cards[0].tier, 1);
        assert_eq!(config.all_cards[6].card.id, 17);
        assert_eq!(config.all_cards[6].tier, 3);

        assert_eq!(config.all_nobles.len(), 3);
        assert_eq!(config.all_nobles[0].id, 1);
    }
}