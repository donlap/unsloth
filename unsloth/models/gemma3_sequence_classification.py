"""
Universal Gemma-3 sequence-classification heads.
Source: https://github.com/huggingface/transformers/issues/36755#issuecomment-2929098303
Works with
  • google/gemma-3-1b-(base|it)  (text only)
  • google/gemma-3-4b-* and larger (multimodal)
Import once, early, then call
    AutoModelForSequenceClassification.from_pretrained(...)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import (
    Gemma3Config, Gemma3nConfig, Gemma3TextConfig, Gemma3nTextConfig,
)
from transformers import (
    Gemma3Model, Gemma3nModel, Gemma3TextModel, Gemma3nTextModel, Gemma3PreTrainedModel, Gemma3nPreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import bitsandbytes as bnb


# ───────────────────── helpers ─────────────────────────────────────────
def _txt(cfg):                              # text sub-config
    return getattr(cfg, "text_config", cfg)


def _pool_last(token_logits, input_ids, pad_id):
    if input_ids is None or pad_id is None:
        return token_logits[:, -1]
    ends = (~input_ids.eq(pad_id)).cumsum(-1).argmax(-1)
    return token_logits[torch.arange(token_logits.size(0)), ends]


def _compute_loss(config, logits, labels, num_labels):
    if labels is None:
        return None
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif labels.dtype in (torch.long, torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        return nn.MSELoss()(logits.squeeze(), labels.squeeze())
    if config.problem_type == "single_label_classification":
        return nn.CrossEntropyLoss()(logits, labels)
    return nn.BCEWithLogitsLoss()(logits, labels)


# ───────────────────── multimodal variant (4 B+) ──────────────────────
class Gemma3ForSequenceClassification(Gemma3Model):
    config_class = Gemma3Config
    _no_split_modules = ["GemmaBlock"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.score = nn.Linear(_txt(config).hidden_size, self.num_labels, bias=False)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        self.post_init()

    def get_output_embeddings(self):
        return self.score

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # Access embeddings through the language model
        if hasattr(self, 'model'):
            embedding_layer = self.model.embed_tokens
        elif hasattr(self, 'language_model'):
            embedding_layer = self.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, pixel_values=None, past_key_values=None,
        labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        token_logits = self.score(outputs.last_hidden_state)
        token_logits = self.dropout(token_logits)
        pad_id = getattr(self.config, "pad_token_id",
                         getattr(_txt(self.config), "pad_token_id", None))
        pooled = _pool_last(token_logits, input_ids, pad_id)
        loss = _compute_loss(self.config, pooled, labels, self.num_labels)

        if not return_dict:
            out = (pooled,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states
                if output_hidden_states else None,
            attentions=outputs.attentions
                if output_attentions else None,
        )


# ───────────────────── text-only variant (1 B) ────────────────────────
class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel):
    """
    **Wraps** Gemma3TextModel in `self.model` – keeps `model.*`
    prefixes so every pretrained weight loads.
    """
    config_class = Gemma3TextConfig
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)
        self.score = nn.Linear(_txt(config).hidden_size, self.num_labels, bias=False)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        self.post_init()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.score

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # Access embeddings through the language model
        if hasattr(self, 'model'):
            embedding_layer = self.model.embed_tokens
        elif hasattr(self, 'language_model'):
            embedding_layer = self.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, past_key_values=None, labels=None,
        use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        token_logits = self.score(outputs.last_hidden_state)
        token_logits = self.dropout(token_logits)
        pooled = _pool_last(token_logits, input_ids, self.config.pad_token_id)
        loss = _compute_loss(self.config, pooled, labels, self.num_labels)

        if not return_dict:
            out = (pooled,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states
                if output_hidden_states else None,
            attentions=outputs.attentions
                if output_attentions else None,
        )


# ───────────────────── Gemma 3n multimodal ──────────────────────
class Gemma3nForSequenceClassification(Gemma3nPreTrainedModel):
    """
    **Wraps** Gemma3nModel in `self.model` – keeps `model.*`
    prefixes so every pretrained weight loads.
    """
    config_class = Gemma3nConfig
    _no_split_modules = ["GemmaBlock"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3nModel(config)

        text_config = getattr(config, "text_config", config)
        self.score = nn.Linear(text_config.hidden_size, self.num_labels, bias=False)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.score

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        embedding_layer = self.model.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None,
        pixel_values=None, past_key_values=None,
        labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        token_logits = self.score(hidden_states)
        token_logits = self.dropout(token_logits)

        pad_id = getattr(self.config, "pad_token_id",
                         getattr(getattr(self.config, "text_config", self.config), "pad_token_id", None))
        pooled = _pool_last(token_logits, input_ids, pad_id)
        loss = _compute_loss(self.config, pooled, labels, self.num_labels)

        if not return_dict:
            out = (pooled,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ───────────────────── Gemma 3n text-only variant (1 B) ────────────────────────
class Gemma3nTextForSequenceClassification(Gemma3nPreTrainedModel):
    """
    **Wraps** Gemma3TextModel in `self.model` – keeps `model.*`
    prefixes so every pretrained weight loads.
    """
    config_class = Gemma3nTextConfig
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3nTextModel(config)
        self.score = nn.Linear(_txt(config).hidden_size, self.num_labels, bias=False)
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        self.post_init()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.score

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        # Access embeddings through the language model
        if hasattr(self, 'model'):
            embedding_layer = self.model.embed_tokens
        elif hasattr(self, 'language_model'):
            embedding_layer = self.language_model.embed_tokens
        self._require_grads_hook = embedding_layer.register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, past_key_values=None, labels=None,
        use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        token_logits = self.score(outputs.last_hidden_state)
        token_logits = self.dropout(token_logits)
        pooled = _pool_last(token_logits, input_ids, self.config.pad_token_id)
        loss = _compute_loss(self.config, pooled, labels, self.num_labels)

        if not return_dict:
            out = (pooled,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states
                if output_hidden_states else None,
            attentions=outputs.attentions
                if output_attentions else None,
        )


# ───────────────────── register with HF factory ───────────────────────
AutoModelForSequenceClassification.register(
    Gemma3Config, Gemma3ForSequenceClassification)
AutoModelForSequenceClassification.register(
    Gemma3TextConfig, Gemma3TextForSequenceClassification)
AutoModelForSequenceClassification.register(
    Gemma3nConfig, Gemma3nForSequenceClassification)
AutoModelForSequenceClassification.register(
    Gemma3nTextConfig, Gemma3nTextForSequenceClassification)
