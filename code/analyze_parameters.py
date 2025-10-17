#!/usr/bin/env python3
"""
Script détaillé pour analyser tous les paramètres de last_param.json
"""
import json
import os

def analyze_parameters():
    """Analyse détaillée de tous les paramètres"""
    
    # Analyse de chaque paramètre avec vérifications spécifiques
    parameter_analysis = {
        'mean_hologram_image_path': {
            'type': str,
            'check': lambda x: os.path.exists(x) if x else True,
            'description': 'Chemin vers l\'hologramme moyen'
        },
        'wavelength': {
            'type': float,
            'check': lambda x: 400e-9 <= float(x) <= 800e-9,
            'description': 'Longueur d\'onde en mètres (400-800nm)'
        },
        'medium_optical_index': {
            'type': float,
            'check': lambda x: 1.0 <= float(x) <= 2.0,
            'description': 'Indice optique du milieu (typiquement 1.33 pour eau)'
        },
        'holo_size_x': {
            'type': int,
            'check': lambda x: 64 <= int(x) <= 4096,
            'description': 'Taille X de l\'hologramme en pixels'
        },
        'holo_size_y': {
            'type': int,
            'check': lambda x: 64 <= int(x) <= 4096,
            'description': 'Taille Y de l\'hologramme en pixels'
        },
        'pixel_size': {
            'type': float,
            'check': lambda x: 1e-6 <= float(x) <= 50e-6,
            'description': 'Taille du pixel en mètres'
        },
        'objective_magnification': {
            'type': float,
            'check': lambda x: 1 <= float(x) <= 100,
            'description': 'Grossissement de l\'objectif'
        },
        'distance_ini': {
            'type': float,
            'check': lambda x: 0 <= float(x) <= 1e-3,
            'description': 'Distance initiale de propagation en mètres'
        },
        'number_of_planes': {
            'type': int,
            'check': lambda x: 1 <= int(x) <= 1000,
            'description': 'Nombre de plans de reconstruction'
        },
        'step': {
            'type': float,
            'check': lambda x: 1e-8 <= float(x) <= 1e-5,
            'description': 'Pas de propagation en mètres'
        },
        'high_pass': {
            'type': int,
            'check': lambda x: 0 <= int(x) <= 500,
            'description': 'Filtre passe-haut en pixels'
        },
        'low_pass': {
            'type': int,
            'check': lambda x: 0 <= int(x) <= 500,
            'description': 'Filtre passe-bas en pixels'
        },
        'focus_type': {
            'type': str,
            'check': lambda x: x in ['TENEGRAD', 'SUM_OF_INTENSITY', 'SUM_OF_LAPLACIAN', 'SUM_OF_VARIANCE'],
            'description': 'Type d\'algorithme de focus'
        },
        'sum_size': {
            'type': int,
            'check': lambda x: 1 <= int(x) <= 50,
            'description': 'Taille de la fenêtre pour le calcul de focus'
        },
        'remove_mean': {
            'type': bool,
            'check': lambda x: isinstance(x, bool),
            'description': 'Suppression de l\'hologramme moyen'
        },
        'cleaning_type': {
            'type': str,
            'check': lambda x: x in ['division', 'subtraction'],
            'description': 'Type de nettoyage de l\'hologramme'
        },
        'batch_threshold': {
            'type': [str, bool],
            'check': lambda x: True,  # Flexible
            'description': 'Seuillage en mode batch'
        },
        'nb_StdVar_threshold': {
            'type': float,
            'check': lambda x: 0.1 <= float(x) <= 1000,
            'description': 'Seuil en nombre d\'écarts-types'
        },
        'connectivity': {
            'type': int,
            'check': lambda x: int(x) in [6, 18, 26],
            'description': 'Connectivité 3D (6, 18 ou 26)'
        },
        'min_voxel': {
            'type': int,
            'check': lambda x: 0 <= int(x) <= 10000,
            'description': 'Nombre minimum de voxels'
        },
        'max_voxel': {
            'type': int,
            'check': lambda x: 0 <= int(x) <= 1000000,
            'description': 'Nombre maximum de voxels'
        },
        'holograms_directory': {
            'type': str,
            'check': lambda x: os.path.exists(x) if x else False,
            'description': 'Répertoire des hologrammes'
        },
        'image_type': {
            'type': str,
            'check': lambda x: x.upper() in ['BMP', 'PNG', 'JPG', 'JPEG', 'TIFF'],
            'description': 'Type d\'image'
        },
        'additional_display': {
            'type': str,
            'check': lambda x: x in ['None', 'Centroid positions', 'Segmentation'],
            'description': 'Affichage additionnel'
        }
    }
    
    try:
        # Load last_param.json from project root (parent of 'code' folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/code
        project_root = os.path.dirname(script_dir)  # /path/to/project
        param_file = os.path.join(project_root, "last_param.json")
        
        with open(param_file, 'r') as f:
            params = json.load(f)

        print("=" * 50)
        
        errors = []
        warnings = []
        
        for param_name, analysis in parameter_analysis.items():
            if param_name in params:
                value = params[param_name]

                print(f"   🏷  Type: {type(value).__name__}")
                
                # Vérification du type
                expected_types = analysis['type'] if isinstance(analysis['type'], list) else [analysis['type']]
                type_ok = any(isinstance(value, t) for t in expected_types)
                
                if not type_ok:
                    # Essai de conversion
                    try:
                        if str in expected_types:
                            converted_value = str(value)
                        elif float in expected_types:
                            converted_value = float(value)
                        elif int in expected_types:
                            converted_value = int(float(value))  # Support "123" -> 123
                        elif bool in expected_types:
                            converted_value = bool(value)
                        else:
                            converted_value = value

                        value = converted_value
                        type_ok = True
                    except Exception as e:
                        errors.append(f" {param_name}: Erreur de type/conversion: {e}")
                        print(f"    Type invalide: attendu {expected_types}, trouvé {type(value)}")
                        continue
                
                # Vérification des contraintes
                try:
                    if analysis['check'](value):
                        pass
                    else:
                        warnings.append(f"  {param_name}: Valeur hors limites: {value}")

                except Exception as e:
                    warnings.append(f"  {param_name}: Erreur de validation: {e}")

            else:
                errors.append(f" {param_name}: MANQUANT")
        
        # Calcul medium_wavelength
        print(f"\n🧮 CALCULS DÉRIVÉS")
        print("=" * 20)
        try:
            wavelength = float(params['wavelength'])
            medium_optical_index = float(params['medium_optical_index'])
            medium_wavelength = wavelength / medium_optical_index
            print(f"medium_wavelength = {wavelength} / {medium_optical_index} = {medium_wavelength:.6e} m")
            print(f"                  = {medium_wavelength * 1e9:.1f} nm")
        except Exception as e:
            errors.append(f" Calcul medium_wavelength impossible: {e}")
        
        # Vérification des chemins
        print(f"\n📁 VÉRIFICATION DES CHEMINS")
        print("=" * 30)
        
        if params.get('mean_hologram_image_path'):
            path = params['mean_hologram_image_path']
            if os.path.exists(path):
                pass
            else:
                warnings.append(f"  Hologramme moyen introuvable: {path}")

        if params.get('holograms_directory'):
            path = params['holograms_directory']
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(params.get('image_type', 'bmp').lower())]

                print(f"    {len(files)} fichiers {params.get('image_type', 'bmp').upper()}")
            else:
                warnings.append(f"  Répertoire introuvable: {path}")

        print("=" * 10)
        print(f" Paramètres OK: {24 - len(errors)}/24")
        print(f" Erreurs: {len(errors)}")
        print(f"  Avertissements: {len(warnings)}")
        
        if errors:

            for error in errors:
                print(f"   {error}")
        
        if warnings:

            for warning in warnings:
                print(f"   {warning}")
        
        if not errors and not warnings:
            pass
        return len(errors) == 0
        
    except FileNotFoundError:

        return False
    except json.JSONDecodeError as e:

        return False
    except Exception as e:

        return False

if __name__ == "__main__":
    analyze_parameters()