import argparse

from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train and compare customer churn models.")
    parser.add_argument("--data", help="Path to a labeled customer churn CSV.")
    args = parser.parse_args()
    print(train(args.data) if args.data else train())


if __name__ == "__main__":
    main()
