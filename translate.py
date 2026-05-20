import pandas as pd
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm

INPUT_DIR      = "data/spanish_subset/"
SPANISH_COLUMN = "text"
ID_COLUMN      = "comment_id"


def setup_translation():
    """Download and install the ES→EN and EN→ES models (runs once)."""
    print("Checking/updating translation package index...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    for from_code, to_code in [("es", "en"), ("en", "es")]:
        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if pkg is None:
            raise RuntimeError(f"{from_code}→{to_code} package not found.")
        print(f"Installing {from_code}→{to_code} model...")
        argostranslate.package.install_from_path(pkg.download())

    print("Translation models ready.\n")


def translate_text(text, translator):
    """Translate a single cell; return original on failure."""
    try:
        if pd.isna(text) or str(text).strip() == "":
            return ""
        return translator.translate(str(text).strip())
    except Exception as e:
        print(f"Could not translate: '{str(text)[:40]}...' → {e}")
        return text


def back_translate_split(split: str) -> None:
    """
    Load <split>.csv, produce <split>_bt.csv with two new columns:
      - english_translation   (ES→EN)
      - backtranslated_text   (ES→EN→ES)

    Only comment_id and backtranslated_text are written to the output file
    so it can be joined back to the main dataset on comment_id.
    """
    input_path  = INPUT_DIR + f"{split}.csv"
    output_path = INPUT_DIR + f"{split}_bt.csv"

    print(f"\n--- Processing {input_path} ---")
    df = pd.read_csv(input_path)
    print(f"{len(df)} rows loaded from '{SPANISH_COLUMN}' column.\n")

    es_en = argostranslate.translate.get_translation_from_codes("es", "en")
    en_es = argostranslate.translate.get_translation_from_codes("en", "es")

    tqdm.pandas(desc=f"[{split}] ES→EN")
    df["english_translation"] = df[SPANISH_COLUMN].progress_apply(
        lambda x: translate_text(x, es_en)
    )

    tqdm.pandas(desc=f"[{split}] EN→ES (back)")
    df["backtranslated_text"] = df["english_translation"].progress_apply(
        lambda x: translate_text(x, en_es)
    )

    df[[ID_COLUMN, "backtranslated_text"]].to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(df[[SPANISH_COLUMN, "english_translation", "backtranslated_text"]].head(3).to_string())


def main():
    setup_translation()
    back_translate_split("train")


if __name__ == "__main__":
    main()
