This is the entry-level code directory for the metaexplainer.

Each of the three sub-stages of the metaexplainer: Decompose, Delegate and Synthesis have their own subdirectories.

The .py files that are stored outside of these directories contain functions that are used commonly by these sub-stages.

All code should be run from this level, i.e., after activating the metaexplainer via ``conda activate metaexplainer''.

For example, you could run: python delegate/parse_machine_interpretation.py.
