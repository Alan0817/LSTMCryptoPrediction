"""Command-line wrapper for the default BTC data-processing pipeline."""

from data_processing import run_data_pipeline


def main():
    data = run_data_pipeline()
    print(data.head())
    print(data.columns)
    print(data.shape)
    print("Processed dataset saved.")


if __name__ == "__main__":
    main()
