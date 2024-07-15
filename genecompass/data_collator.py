from collections.abc import Mapping
from typing import List, Union, Dict, Any, Tuple, Optional
from transformers.data.data_collator import _torch_collate_batch
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorForLanguageModelingModified(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            # batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            input_ids = torch.tensor([example['input_ids'] for example in examples])
            values = torch.tensor([example['values'] for example in examples])
            if 'species' in examples[0].keys():
                species= torch.tensor([example['species'] for example in examples]) 
            else:
                species= None
            original_lengths = (input_ids != 0).sum(dim=1)
            attention_masks = torch.zeros_like(input_ids)
            for i, length in enumerate(original_lengths):
                attention_masks[i, :length] = 1
            batch = {'input_ids': input_ids, 'values': values, 'attention_mask': attention_masks, 'species':species}

        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["values"], batch["labels"], batch["labels_values"] = self.torch_mask_tokens(
                batch["input_ids"], batch["values"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, values: Any,special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        labels_values = values.clone()

        # TODO: 这一行在外面实现了
        # special_tokens_mask = torch.where(inputs==0, 1, 0).to(inputs.device)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        values[indices_replaced] = -1

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        values[indices_random] = -1

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, values, labels, labels_values
