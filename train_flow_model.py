import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

def train_flow_model():
    print("Flow Training (separate script) ...")

    # Charger le dataset
    flow_dataset = pd.read_csv('FlowStatsfile.csv')

    # Nettoyage des colonnes spécifiques
    flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
    flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
    flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

    # Préparer les features et les labels
    X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
    y_flow = flow_dataset.iloc[:, -1].values

    # Diviser les données en ensembles d'entraînement et de test
    X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(
        X_flow, y_flow, test_size=0.25, random_state=0
    )

    # Entraîner le modèle
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_flow_train, y_flow_train)

    # Évaluer le modèle
    y_flow_pred = classifier.predict(X_flow_test)
    cm = confusion_matrix(y_flow_test, y_flow_pred)
    acc = accuracy_score(y_flow_test, y_flow_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Failure Rate: {:.2f}%".format((1 - acc) * 100))

    # Enregistrer le modèle
    joblib.dump(classifier, 'flow_model.pkl')
    print("Model saved as 'flow_model.pkl'")

if __name__ == "__main__":
    train_flow_model()
