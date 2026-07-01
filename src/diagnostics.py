from __future__ import annotations

from src.db import load_dataset_diagnostics


def main() -> None:
    diagnostics = load_dataset_diagnostics()
    print("Volleyball dataset diagnostics")
    for row in diagnostics.itertuples(index=False):
        print(f"{row.metric}: {row.value}")


if __name__ == "__main__":
    main()
