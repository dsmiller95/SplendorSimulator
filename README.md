# SplendorSimulator
A simulation of the board game splendor written in python. Will attempt to train an AI to solve the game.

## Setup

### python training and webserver
Install python 3.7-3.11 . python 3.12 is not supported.

Setup:
```bash
source setup_windows.sh
```

Run:
```bash
python main.py
```

### frontend

Install npm

run commands:
```bash
cd splendor-web-viewer
npm install
```
to dev locally:
```bash
npm run start
```

## Develop

### rust submodule

Prerequisites:
- setup python training and webserver
- configure rust defaults for the directory using using `setup_rust.sh` . must be run once per machine.
- rust is installed via rustup https://www.rust-lang.org/tools/install


Setup + develop:
```bash
source rust_dev.sh
```

Run tests on the python binding. Will build the rust project
```bash
source rust_test.sh
```