"""Master runner that executes all analysis scripts.

Each individual ``run_*.py`` module defines a ``main`` function that performs a
specific analysis (e.g., token length, frequency ratios, readability, etc.).
This script imports those modules and invokes their ``main`` functions in a
deterministic order so that a single command runs the full suite of analyses.
"""

# Import the individual runner modules. The imports are placed at the top so
# that any import‑time errors are raised early, making debugging easier.
from features import run_token_length, run_freq, run_readability, run_sent_length, \
    run_subjunctive, run_negations


def main() -> None:
    run_token_length.main()
    run_sent_length.main()
    run_readability.main()
    run_freq.main()
    run_negations.main()
    run_subjunctive.main()

if __name__ == "__main__":
    main()