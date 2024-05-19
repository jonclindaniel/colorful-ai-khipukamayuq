from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, BertTokenizerFast, BertForMaskedLM, BertConfig, DataCollatorForLanguageModeling
import optuna

SEED = 7
FILE_PATH = "./data"
MODEL_PATH = "./pretrained-bert"

def model_init(trial):
    model_config = BertConfig(max_position_embeddings=512)
    model = BertForMaskedLM(config=model_config)
    return model


def hp_space(trial):
    # try multiple seeds and learning rates and
    # select learning rate that is best across all three seeds
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-5, step=1e-5),
        "seed": trial.suggest_int("seed", SEED - 1, SEED + 1, step=1)
    }


if __name__ == "__main__":
    # load data
    train_dataset = load_from_disk(f"{FILE_PATH}/train")
    test_dataset = load_from_disk(f"{FILE_PATH}/test")

    # load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

    # initialize the data collator, randomly masking (default 15%) of the tokens for
    # the Masked Language Modeling (MLM) task
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        max_steps=400,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
    )

    # initialize the trainer and pass everything to it
    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # search for optimal learning rate across three seeds
    trainer.hyperparameter_search(hp_space=hp_space,
                                  backend="optuna",
                                  sampler=optuna.samplers.BruteForceSampler()
    )
    # Output: 5e-5 is the best performing learning rate across all three seeds