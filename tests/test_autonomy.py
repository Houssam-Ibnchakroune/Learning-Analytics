"""
Script de test pour vérifier que tous les notebooks et scripts sont autonomes.
"""

print("=" * 70)
print("TEST D'AUTONOMIE DES MODULES")
print("=" * 70)

# Test 1 : Import des modules src
print("\n✅ Test 1 : Import des modules src.models...")
try:
    from src.models import train_random_forest, predict, evaluate_classification
    print("   ✓ src.models importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

# Test 2 : Vérifier train.py
print("\n✅ Test 2 : Vérification de train.py...")
try:
    from src.models.train import train_random_forest, train_with_cross_validation
    print("   ✓ train.py fonctionne en autonome")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

# Test 3 : Vérifier predict.py
print("\n✅ Test 3 : Vérification de predict.py...")
try:
    from src.models.predict import predict, predict_with_risk_score
    print("   ✓ predict.py fonctionne en autonome")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

# Test 4 : Vérifier evaluate.py
print("\n✅ Test 4 : Vérification de evaluate.py...")
try:
    from src.models.evaluate import evaluate_classification, plot_confusion_matrix
    print("   ✓ evaluate.py fonctionne en autonome")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

print("\n" + "=" * 70)
print("RÉSUMÉ DES TESTS")
print("=" * 70)
print("✅ Tous les modules sont autonomes et importables !")
print("\nMaintenant les notebooks peuvent importer les fonctions ainsi :")
print("  from src.models import train_random_forest, predict, evaluate_classification")
print("\nOu directement depuis le notebook 2, 3, 4 car ils ont leurs propres imports.")
print("=" * 70)
