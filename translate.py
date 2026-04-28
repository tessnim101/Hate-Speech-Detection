import pandas as pd
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm

INPUT_FILE        = "data/train.csv"    
OUTPUT_FILE       = "data/train_translated.csv" 
SPANISH_COLUMN    = "text"     
TRANSLATED_COLUMN = "english_translation" 

def setup_translation():
    """Download and install the Spanish → English model (runs once)."""
    print("Checking translation model...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    pkg = next(
        (p for p in available if p.from_code == "es" and p.to_code == "en"),
        None
    )

    if pkg is None:
        raise RuntimeError("Spanish → English package not found.")

    print("Installing Spanish → English model...")
    argostranslate.package.install_from_path(pkg.download())
    print("Model ready.\n")


def translate_text(text, translator):
    """Translate a single cell, return original if it fails."""
    try:
        if pd.isna(text) or str(text).strip() == "":
            return ""
        return translator.translate(str(text).strip())
    except Exception as e:
        print(f"Could not translate: '{str(text)[:40]}...' → {e}")
        return text 


def main():
    # 1. Setup model
    setup_translation()

    # 2. Load translator
    translator = argostranslate.translate.get_translation_from_codes("es", "en")

    # 3. Load file
    print(f"Loading file: {INPUT_FILE}")
    if INPUT_FILE.endswith(".csv"):
        df = pd.read_csv(INPUT_FILE)
    elif INPUT_FILE.endswith((".xlsx", ".xls")):
        df = pd.read_excel(INPUT_FILE)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    print(f"{len(df)} rows found in column '{SPANISH_COLUMN}'\n")

    # 4. Translate with progress bar
    tqdm.pandas(desc="Translating")
    df[TRANSLATED_COLUMN] = df[SPANISH_COLUMN].progress_apply(
        lambda x: translate_text(x, translator)
    )

    # 5. Save output
    if OUTPUT_FILE.endswith(".csv"):
        df.to_csv(OUTPUT_FILE, index=False)
    elif OUTPUT_FILE.endswith((".xlsx", ".xls")):
        df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n Done! Saved to: {OUTPUT_FILE}")
    print(df[[SPANISH_COLUMN, TRANSLATED_COLUMN]].head(5)) 


if __name__ == "__main__":
    main()