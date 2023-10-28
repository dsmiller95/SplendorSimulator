# SplendorSimulator
A simulation of the board game splendor written in python. Will attempt to train an AI to solve the game.

## Setup

### python training and webserver
Install python 3.7+

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

## Develop

### rust submodule

Must setup python training and webserver

Must be using rust nightly, can be configured using `setup_rust.sh` . must be run once per machine.

Setup + develop:
```bash
source rust_dev.sh
```

Run tests on the python binding. Will build the rust project
```bash
source rust_test.sh
```