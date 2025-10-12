from sklearn.tree import DecisionTreeRegressor
import misc

def main():
    df = misc.load_data()
    X_train, X_test, y_train, y_test = misc.preprocess_data(df)
    model = misc.train_model(X_train, y_train, DecisionTreeRegressor(random_state=42))
    mse = misc.evaluate_model(model, X_test, y_test)
    print(f"Decision Tree - Average MSE on Test Set: {mse:.2f}")

if __name__ == "__main__":
    main()
