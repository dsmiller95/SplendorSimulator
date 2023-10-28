use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rust_splendor;
use rust_splendor::constants::{CardPickInTier, CardPickOnBoard, CardTier, OpenCardPickInTier, PlayerCardPick, ReservedCardSelection};
use rust_splendor::game_actions::turn::Turn;
use crate::components::resource_type::SplendorResourceType;

#[pyclass]
pub struct SplendorTurn {
    wrapped: Turn,
}


#[pymethods]
impl SplendorTurn {
    #[classmethod]
    fn new_take_three(_cls: &PyType, first: SplendorResourceType, second: SplendorResourceType, third: SplendorResourceType) -> PyResult<Self> {
        let turn = Turn::TakeThreeTokens(
                first.try_into()?,
                second.try_into()?,
                third.try_into()?,);

        if !turn.validate() {
            return Err(PyValueError::new_err("invalid turn"));
        }

        Ok(SplendorTurn{
            wrapped: turn
        })
    }

    #[classmethod]
    fn new_take_two(_cls: &PyType, single: SplendorResourceType) -> PyResult<Self> {
        Ok(SplendorTurn {
            wrapped: Turn::TakeTwoTokens(single.try_into()?),
        })
    }

    /// pick is 0..4. 0 is the hidden card, 1..3 are exposed cards
    /// tier is 1..3
    #[classmethod]
    fn new_buy_card_on_board(_cls: &PyType, tier: u8, pick: u8) -> PyResult<Self> {
        let tier = match tier {
            1 => CardTier::CardTier1,
            2 => CardTier::CardTier2,
            3 => CardTier::CardTier3,
            _ => return Err(PyValueError::new_err(format!("Invalid tier {}", tier))),
        };
        let pick = match pick {
            0 => CardPickInTier::HiddenCard,
            1 => CardPickInTier::OpenCard(OpenCardPickInTier::OpenCardPickInTier1),
            2 => CardPickInTier::OpenCard(OpenCardPickInTier::OpenCardPickInTier2),
            3 => CardPickInTier::OpenCard(OpenCardPickInTier::OpenCardPickInTier3),
            4 => CardPickInTier::OpenCard(OpenCardPickInTier::OpenCardPickInTier4),
            _ => return Err(PyValueError::new_err(format!("invalid pick in tier {}", pick))),
        };
        Ok(SplendorTurn {
            wrapped: Turn::PurchaseCard(PlayerCardPick::OnBoard(CardPickOnBoard{
                tier,
                pick,
            })),
        })
    }
    /// reserved is 1..3
    #[classmethod]
    fn new_buy_reserved_card(_cls: &PyType, reserved_pick: u8) -> PyResult<Self> {
        let reserved_pick = match reserved_pick {
            1 => ReservedCardSelection::ReservedCardSelection1,
            2 => ReservedCardSelection::ReservedCardSelection2,
            3 => ReservedCardSelection::ReservedCardSelection3,
            _ => return Err(PyValueError::new_err(format!("invalid pick in tier {}", reserved_pick))),
        };
        Ok(SplendorTurn {
            wrapped: Turn::PurchaseCard(PlayerCardPick::Reserved(reserved_pick)),
        })
    }
}