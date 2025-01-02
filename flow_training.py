from sklearn.neighbors import KNeighborsClassifier
import joblib

def flow_training(self):
    self.logger.info("Flow Training (loading pre-trained model) ...")

    try:
        # Charger le modèle enregistré
        self.flow_model = joblib.load('flow_model.pkl')
        self.logger.info("Model successfully loaded from 'flow_model.pkl'")
    except FileNotFoundError:
        self.logger.error("Model file 'flow_model.pkl' not found. Please train the model separately.")
        return
    except Exception as e:
        self.logger.error(f"An error occurred while loading the model: {e}")
        return

    self.logger.info("Model is now loaded and ready to use.")
