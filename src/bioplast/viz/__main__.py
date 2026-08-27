"""CLI визуализации, локального сервера и сквозного smoke-теста."""

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "serve":
        from bioplast.viz.server import main as serve_main

        return serve_main(args[1:])
    if args and args[0] == "smoke":
        from bioplast.viz.smoke import main as smoke_main

        return smoke_main(args[1:])

    # Preserve the original plotting CLI:
    # `uv run python -m bioplast.viz runs --all`.
    from bioplast.viz.plot import main as plot_main

    return plot_main(args)

if __name__ == "__main__":
    sys.exit(main())
