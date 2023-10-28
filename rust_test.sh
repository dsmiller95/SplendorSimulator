sh << END
set -e # exit script on first failures
cd rust_module
cargo test
cd bindings
maturin develop
cd ../..
cd python_rust_bind_adapter
pytest
cd ..
END