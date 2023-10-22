use pyo3::prelude::*;
use rust_splendor;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
/// references a function in a different crate in the workspace.
#[pyfunction]
fn add_one_as_string(a: i32) -> PyResult<TestStruct> {
    let res = TestStruct{
        something: rust_splendor::add_one(a).to_string(),
        another: a
    };
    Ok(res)
}

#[pyclass(get_all)]
struct TestStruct {
    pub something: String,
    another: i32
}

/// A Python module implemented in Rust.
#[pymodule]
fn splendor_simulation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(add_one_as_string, m)?)?;
    Ok(())
}