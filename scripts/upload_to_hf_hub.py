"""Upload the three trained model checkpoints to the Hugging Face Hub.

Run once after creating an HF account and logging in via huggingface-cli.
The script creates three model repositories and uploads each model
together with its companion files (vocab, tokenizer, etc.).
"""
from pathlib import Path

from huggingface_hub import HfApi, create_repo

HF_USERNAME = "sandraFogang"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints"

REPOS = {
    f"{HF_USERNAME}/nlp-sentiment-tfidf": [
        ("tfidf_uni_bi_sublinear.pt", "classifier.pt"),
        ("tfidf_uni_bi_sublinear_vectorizer.pkl", "vectorizer.pkl"),
    ],
    f"{HF_USERNAME}/nlp-sentiment-bilstm": [
        ("bilstm_v2_full_glove300.pt", "classifier.pt"),
        ("bilstm_v2_full_glove300_vocab.pkl", "vocab.pkl"),
    ],
    f"{HF_USERNAME}/nlp-sentiment-distilbert": [
        ("distilbert_full_finetune.pt", "classifier.pt"),
    ],
}

DISTILBERT_TOKENIZER_DIR = CHECKPOINTS / "distilbert_tokenizer"


def main() -> None:
    api = HfApi()

    for repo_id, files in REPOS.items():
        print(f"\nCreating or updating repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        for source_name, target_name in files:
            source = CHECKPOINTS / source_name
            if not source.exists():
                print(f"  SKIP (missing): {source}")
                continue

            print(f"  Uploading {source.name} -> {target_name}")
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=target_name,
                repo_id=repo_id,
                repo_type="model",
            )

        if "distilbert" in repo_id and DISTILBERT_TOKENIZER_DIR.exists():
            print(f"  Uploading tokenizer folder")
            api.upload_folder(
                folder_path=str(DISTILBERT_TOKENIZER_DIR),
                path_in_repo="tokenizer",
                repo_id=repo_id,
                repo_type="model",
            )

    print("\nAll models uploaded.")


if __name__ == "__main__":
    main()