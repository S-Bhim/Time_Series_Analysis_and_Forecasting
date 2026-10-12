import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Climate time series research pipeline")
    parser.add_argument("command", choices=["run", "serve", "audit"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "run":
        from .pipeline import run
        run(args.root)
    elif args.command == "audit":
        import json
        from .data import load_data
        print(json.dumps(load_data(args.root/"data/raw/monthly_csv.csv")[2], indent=2))
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
