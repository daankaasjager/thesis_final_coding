import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


@rank_zero_only
def print_batch(train_ds, valid_ds, tokenizer, k=8):
    """Prints the first and last k tokens of the first batch of the train and valid dataloaders."""
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        logger.info(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        logger.info("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        logger.info(
            f"First {k} tokens:", tokenizer.decode(first, skip_special_tokens=True)
        )
        logger.info("ids:", first)
        logger.info(
            f"Last {k} tokens:", tokenizer.decode(last, skip_special_tokens=False)
        )
        logger.info("ids:", last)
