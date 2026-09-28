"""Puts the project root on sys.path so tests/ can `import planner`.

pytest prepends the directory holding the root conftest.py, which is what makes the plain
`import planner` in tests/test_planner.py resolve. Nothing else needed here.
"""
