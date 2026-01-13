"""
Script de rollback de modèle avec MLflow
Usage: python scripts/rollback_model.py <version_number>
"""
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import shutil
import os
import sys

def rollback_to_version(model_name: str, target_version: int):
    """Rollback vers une version spécifique du modèle"""
    client = MlflowClient()
    
    try:
        # 1. Vérifier que la version existe
        model_version = client.get_model_version(model_name, target_version)
        print(f"📦 Version {target_version} trouvée: {model_version.current_stage}")
        
        # 2. Récupérer le modèle de la version cible
        model_uri = f"models:/{model_name}/{target_version}"
        print(f"⬇️  Téléchargement du modèle depuis {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        
        # 3. Sauvegarder l'ancien modèle
        model_path = "models/best_model.joblib"
        if os.path.exists(model_path):
            backup_path = f"models/best_model.joblib.backup"
            shutil.copy(model_path, backup_path)
            print(f"💾 Ancien modèle sauvegardé dans {backup_path}")
        
        # 4. Charger le nouveau modèle
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Modèle version {target_version} chargé dans {model_path}")
        
        # 5. Mettre à jour le stage dans MLflow (si Model Registry est utilisé)
        try:
            # Archiver la version actuelle en production
            current = client.get_latest_versions(model_name, stages=["Production"])
            if current:
                client.transition_model_version_stage(
                    name=model_name,
                    version=current[0].version,
                    stage="Archived"
                )
                print(f"📦 Version {current[0].version} archivée")
            
            # Promouvoir la version cible
            client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )
            print(f"🚀 Version {target_version} promue en Production")
        except Exception as e:
            print(f"⚠️  Note: Model Registry non configuré ({e})")
            print("   Le modèle local a été mis à jour, mais pas le registry MLflow")
        
        print(f"\n✅ Rollback vers version {target_version} effectué avec succès!")
        print("   Redémarrez l'API pour utiliser le nouveau modèle.")
        
    except Exception as e:
        print(f"❌ Erreur lors du rollback: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rollback_model.py <version_number>")
        print("Exemple: python scripts/rollback_model.py 1")
        sys.exit(1)
    
    try:
        version = int(sys.argv[1])
        model_name = sys.argv[2] if len(sys.argv) > 2 else "iris-model"
        rollback_to_version(model_name, version)
    except ValueError:
        print("❌ Erreur: Le numéro de version doit être un entier")
        sys.exit(1)
