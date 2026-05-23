import json
import os
from pathlib import Path
from enum import Enum, auto

# Repository layout
REPO_ROOT = Path(__file__).resolve().parent.parent

# Import phase modules
import sys, pathlib
# Ensure repository root is on sys.path for absolute imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
from solution import phase1, phase2, phase3, phase4

class State(Enum):
    PHASE1 = auto()
    PHASE2 = auto()
    PHASE3 = auto()
    PHASE4 = auto()
    DONE = auto()

def run_state_machine():
    state = State.PHASE1
    while state != State.DONE:
        if state == State.PHASE1:
            print("Running Phase 1: Chunking KB documents")
            phase1.build_chunks()
            state = State.PHASE2
        elif state == State.PHASE2:
            print("Running Phase 2: TF‑IDF Retrieval")
            phase2.retrieve()
            state = State.PHASE3
        elif state == State.PHASE3:
            print("Running Phase 3: Answer Generation")
            phase3.main()
            state = State.PHASE4
        elif state == State.PHASE4:
            print("Running Phase 4: Evaluation & Grounding Check")
            phase4.main()
            state = State.DONE
    print("All phases completed successfully.")

if __name__ == "__main__":
    run_state_machine()
