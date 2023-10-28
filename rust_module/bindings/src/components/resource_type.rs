
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

pub struct ResourceTokenToTypeConversionError(ResourceTokenType);

impl From<ResourceTokenToTypeConversionError> for PyErr {
    fn from(value: ResourceTokenToTypeConversionError) -> Self {
        use pyo3::exceptions::PyValueError;
        PyValueError::new_err(format!("cannot convert {:?} into a resource token", value.0))
    }
}


impl TryInto<ResourceType> for SplendorResourceType {
    type Error = ResourceTokenToTypeConversionError;

    fn try_into(self) -> Result<ResourceType, Self::Error> {
        let token : ResourceTokenType = self.into();
        match token {
            ResourceTokenType::CostType(resource_type) => Ok(resource_type),
            token => Err(ResourceTokenToTypeConversionError(token)),
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


#[cfg(test)]
mod test{

    use super::*;
    #[test]
    fn test_conversion_error_converts() {
        pyo3::prepare_freethreaded_python();

        let gold_type = SplendorResourceType::Gold;
        fn do_convert(resource_type: SplendorResourceType) -> Result<ResourceType, PyErr> {
            let token : ResourceType = resource_type.try_into()?;
            Ok(token)
        }

        let token = do_convert(gold_type);
        assert!(token.is_err());
        let err = token.unwrap_err();
        assert_eq!(err.to_string(), "ValueError: cannot convert Gold into a resource token");
    }
}