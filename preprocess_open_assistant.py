from datasets import load_dataset
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class PromptResponseExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def process_tree(group):
            prompt_df = group[group["role"] == "prompter"]
            assistant_df = group[group["role"] == "assistant"]

            prompts = prompt_df["text"].tolist()
            responses = (
                assistant_df.groupby("parent_id")["text"].apply(" ".join).tolist()
            )

            if not prompts or len(prompts) != len(responses):
                return pd.DataFrame(columns=["prompt", "response"])

            base_prompt = prompts[0]
            augmented_prompts = [base_prompt] + [
                f"{base_prompt} {prompt}" for prompt in prompts[1:]
            ]

            return pd.DataFrame({"prompt": augmented_prompts, "response": responses})

        return X.groupby("message_tree_id").apply(process_tree).reset_index(drop=True)


def create_preprocessing_pipeline():
    return Pipeline(
        [
            ("prompt_response_extractor", PromptResponseExtractor()),
        ]
    )


# Usage
def preprocess_data(df):
    pipeline = create_preprocessing_pipeline()
    pipeline = pipeline.fit_transform(df)
    final_df = df[df["response"] != ""]
    return final_df


def get_train_val_ds():
    dataset = load_dataset("OpenAssistant/oasst1")

    train_dataset = dataset["train"]  # len(train)=84437 (95%)
    val_dataset = dataset["validation"]  # len(val)=4401 (5%)

    train_df = train_dataset.to_pandas()
    val_df = val_dataset.to_pandas()

    train_ds = preprocess_data(train_df)
    val_ds = preprocess_data(val_df)
    return train_ds, val_ds
