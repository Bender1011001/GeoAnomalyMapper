import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, nargs=1)
    
    print(f"Args: {sys.argv[1:]}")
    try:
        args = parser.parse_args()
        print(f"Parsed region: {args.region[0]}")
    except SystemExit as e:
        print(f"Argparse failed with code {e}")

if __name__ == "__main__":
    main()