"""Re-run the tutorial notebooks and save their outputs for the docs.

ReadTheDocs does not execute the notebooks; it shows the outputs saved in them. Run this after
changing any code cell (text-only edits don't need it):

    python docs/run_tutorials.py                     # all notebooks in docs/tutorials
    python docs/run_tutorials.py docs/tutorials/simulation_workflow.ipynb

Each notebook runs with docs/tutorials as its working directory. Local paths to the installed
package (e.g. C:\\Users\\you\\...\\isotomics) are replaced with <isotomics> in the saved outputs.
"""
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import isotomics

TUTORIALS = Path(__file__).resolve().parent / "tutorials"


def main(paths):
    pkg_dir = str(Path(isotomics.__file__).parent)
    # Match the package path as printed (C:\a\b) and as JSON-escaped (C:\\a\\b)
    scrubs = [re.compile(re.escape(p), re.IGNORECASE) for p in (pkg_dir, pkg_dir.replace("\\", "\\\\"))]
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    failed = False
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        print(f"Running {path.name} ...", flush=True)
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()

        text = nbformat.writes(nb)
        for pattern in scrubs:
            text = pattern.sub(lambda m: "<isotomics>", text)
        nb = nbformat.reads(text, as_version=4)
        nbformat.write(nb, path)

        errors = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"
                  for o in c.outputs if o.output_type == "error"]
        if errors:
            failed = True
            print(f"  errors in cells {errors}")
        else:
            print("  ok")
    return 1 if failed else 0


if __name__ == "__main__":
    args = [Path(p) for p in sys.argv[1:]] or sorted(TUTORIALS.glob("*.ipynb"))
    sys.exit(main(args))
