from src.data.extract import load_oulad_data, get_dataset_info

print("Test du module extract.py\n")
    
try:
        data = load_oulad_data('data/raw/open+university+learning+analytics+dataset/')
        get_dataset_info(data)
        
        print("\nModule extract.py fonctionne correctement!")
        
except Exception as e:
        print(f"\nErreur: {e}")