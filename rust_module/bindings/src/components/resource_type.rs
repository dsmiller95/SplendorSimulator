use pyo3::prelude::*;
use rust_splendor;
use rust_splendor::constants::{ResourceTokenType, ResourceType};

#[pyclass]
#[derive(Clone)]
pub enum SplendorResourceType {
    Emerald,
    Sapphire,
    Ruby,
    Diamond,
    Onyx,
    Gold,
}

impl TryInto<ResourceType> for SplendorResourceType {
    type Error = ResourceTokenType;

    fn try_into(self) -> Result<ResourceType, Self::Error> {
        let token : ResourceTokenType = self.into();
        match token {
            ResourceTokenType::CostType(resource_type) => Ok(resource_type),
            token => Err(token),
        }
    }
}

impl Into<ResourceTokenType> for SplendorResourceType {
    fn into(self) -> ResourceTokenType {
        match self {
            SplendorResourceType::Emerald => ResourceTokenType::CostType(ResourceType::Emerald),
            SplendorResourceType::Sapphire => ResourceTokenType::CostType(ResourceType::Sapphire),
            SplendorResourceType::Ruby => ResourceTokenType::CostType(ResourceType::Ruby),
            SplendorResourceType::Diamond => ResourceTokenType::CostType(ResourceType::Diamond),
            SplendorResourceType::Onyx => ResourceTokenType::CostType(ResourceType::Onyx),
            SplendorResourceType::Gold => ResourceTokenType::Gold,
        }
    }
}