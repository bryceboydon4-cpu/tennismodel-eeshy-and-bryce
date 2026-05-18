from src.features.build_features import build_features
from src.models.train_xgb import train_model
from src.processing.process_matches import process_matches


def main():
    print("Tennis Prediction Engine")
    print("-------------------------")
    print("1. Processing historical matches and Elo ratings")
    process_matches()

    print("\n2. Creating ML training data")
    build_features()

    print("\n3. Training XGBoost model")
    train_model()


if __name__ == "__main__":
    main()
