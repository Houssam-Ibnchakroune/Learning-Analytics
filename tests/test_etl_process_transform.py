from src.data.extract import load_oulad_data
from src.data.transform import prepare_dataset
    
print("Test du module transform.py\n")

try:
    # Charger les données
    data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/')
    
    # Préparer le dataset
    final_df = prepare_dataset(data)
    
    print(f"\nDataset final:")
    print(f"   - Dimensions: {final_df.shape}")
    print(f"   - Colonnes: {list(final_df.columns)}")
    print(f" \n\nhead : {final_df.head()}")
    print(f"\nModule transform.py fonctionne correctement!")
    
except Exception as e:
    print(f"\nErreur: {e}")