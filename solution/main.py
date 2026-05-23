import sys
import pathlib
from enum import Enum

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from solution import phase1, phase2, phase3, phase4
from solution.validate import validate_artifacts


class PipelineState(Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


def transition(state: PipelineState) -> PipelineState:
    print(state.value)
    return state


def run_pipeline():
    transition(PipelineState.INIT)

    # Phase 1 – load + chunk documents
    phase1.build_chunks()
    transition(PipelineState.DOCUMENTS_LOADED)
    transition(PipelineState.DOCUMENTS_CHUNKED)

    # Phase 2 – build TF-IDF index + retrieve
    phase2.retrieve()
    transition(PipelineState.INDEX_BUILT)
    transition(PipelineState.RETRIEVAL_COMPLETE)

    # Phase 3 – generate answers
    phase3.main()
    transition(PipelineState.ANSWERS_GENERATED)

    # Phase 4 – evaluate + grounding check
    phase4.main()
    transition(PipelineState.EVALUATION_COMPLETE)

    # Validate all artifacts
    validate_artifacts()
    transition(PipelineState.VALIDATION_COMPLETE)

    transition(PipelineState.RESULTS_FINALISED)


if __name__ == "__main__":
    run_pipeline()
