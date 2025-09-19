#!/usr/bin/env python3
"""
Script de nettoyage du projet HOLOTRACKER
Supprime les fichiers obsolètes et inutiles
"""
import os
import shutil

# Répertoire racine du projet
PROJECT_ROOT = r"C:\TRAVAIL\RepositoriesGithub\HOLOTRACKER_FULL_PYTHON"
CODE_DIR = os.path.join(PROJECT_ROOT, "code")

# Fichiers à supprimer
FILES_TO_DELETE = [
    # Anciennes architectures
    "controller.py",
    "controller_backup.py", 
    "core_communicator_backup.py",
    "core_communicator_clean.py",
    
    # Anciennes interfaces
    "gui.py",
    "GUI_2.py",
    "hologram_processor.py",
    "message_manager.py",
    
    # Tests obsolètes
    "test_auto_updates.py",
    "test_direct_timing.py",
    "test_display_options.py", 
    "test_info_tab.py",
    "test_integration.py",
    "test_timing_display.py",
    "test_visualization.py",
    "validate_system.py",
    
    # Fichiers temporaires
    "test_interact.ipynb",
    "CONSOLIDATION_SUMMARY.md",
]

# Fichiers racine à supprimer
ROOT_FILES_TO_DELETE = [
    "result_python_sum15_TENEGRAD_STD15_each.csv",
    "result_python_sum15_TENEGRAD_STD15_each_wo_alloc.csv",
    "list_files.py",
]

def cleanup_project():
    """Nettoie le projet en supprimant les fichiers obsolètes"""
    total_size_saved = 0
    files_deleted = 0
    
    print("🧹 NETTOYAGE DU PROJET HOLOTRACKER")
    print("=" * 50)
    
    # Supprimer fichiers dans /code
    for filename in FILES_TO_DELETE:
        filepath = os.path.join(CODE_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            os.remove(filepath)
            total_size_saved += size
            files_deleted += 1
            print(f"✅ Supprimé: code/{filename} ({size:,} bytes)")
        else:
            print(f"⚠️  Pas trouvé: code/{filename}")
    
    # Supprimer fichiers racine
    for filename in ROOT_FILES_TO_DELETE:
        filepath = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            os.remove(filepath)
            total_size_saved += size
            files_deleted += 1
            print(f"✅ Supprimé: {filename} ({size:,} bytes)")
        else:
            print(f"⚠️  Pas trouvé: {filename}")
    
    # Supprimer __pycache__
    pycache_dir = os.path.join(CODE_DIR, "__pycache__")
    if os.path.exists(pycache_dir):
        size_before = sum(os.path.getsize(os.path.join(pycache_dir, f)) 
                         for f in os.listdir(pycache_dir) if os.path.isfile(os.path.join(pycache_dir, f)))
        shutil.rmtree(pycache_dir)
        total_size_saved += size_before
        files_deleted += len(os.listdir(pycache_dir)) if os.path.exists(pycache_dir) else 0
        print(f"✅ Supprimé: __pycache__/ ({size_before:,} bytes)")
    
    # Supprimer .ipynb_checkpoints
    checkpoints_dir = os.path.join(CODE_DIR, ".ipynb_checkpoints")
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        print(f"✅ Supprimé: .ipynb_checkpoints/")
    
    print("=" * 50)
    print(f"🎯 RÉSULTAT:")
    print(f"   Fichiers supprimés: {files_deleted}")
    print(f"   Espace libéré: {total_size_saved:,} bytes ({total_size_saved/1024:.1f} KB)")
    print("")
    print("✅ Nettoyage terminé !")
    
    # Lister les fichiers restants
    print("\n📋 FICHIERS RESTANTS (Architecture actuelle):")
    remaining_files = []
    for file in os.listdir(CODE_DIR):
        if os.path.isfile(os.path.join(CODE_DIR, file)) and file.endswith('.py'):
            remaining_files.append(file)
    
    remaining_files.sort()
    for file in remaining_files:
        print(f"   ✅ {file}")

if __name__ == "__main__":
    response = input("❓ Voulez-vous vraiment nettoyer le projet ? (y/N): ")
    if response.lower() in ['y', 'yes', 'oui']:
        cleanup_project()
    else:
        print("❌ Nettoyage annulé")
