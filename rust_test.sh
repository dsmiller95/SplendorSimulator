cd rust_module/bindings
maturin develop
cd ../..
cd python_rust_bind_adapter
pytest
cd ..