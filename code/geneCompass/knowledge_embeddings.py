from typing import Optional

import torch
import torch.nn as nn

from collections import OrderedDict

import ipdb
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 255):
        super().__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        x = x.float().cuda()
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PriorEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_model)
        # self.activation = nn.ReLU()
        # self.linear2 = nn.Linear(d_model, d_model)
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        # x = self.activation(x)
        # x = self.linear2(x)
        # x = self.norm(x)
        return x


class KnowledgeBertEmbeddings(nn.Module):
    def __init__(self, config, knowledges):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.use_values = config.to_dict().get('use_values', False)
        self.use_promoter = config.to_dict().get('use_promoter', False)
        self.use_co_exp = config.to_dict().get('use_co_exp', False)
        self.use_gene_family = config.to_dict().get('use_gene_family', False)
        self.use_peca_grn = config.to_dict().get('use_peca_grn', False)

        self.concat_values_first = config.to_dict().get('concat_values_first', False)
        self.use_value_emb = config.to_dict().get('use_value_emb', False)

        if self.use_value_emb:
            self.values_embeddings = ContinuousValueEncoder(config.hidden_size)
            self.value_dim = config.hidden_size
        else:
            self.value_dim = 1

        num_hidden = 0
        if self.use_promoter:
            if knowledges.get('promoter', None) is not None:
                self.register_buffer('promoter_knowledge', knowledges['promoter'])
                self.homologous_gene_human2mouse = knowledges['homologous_gene_human2mouse']
                token_num = knowledges['promoter'].size(0)
                self.promoter_embeddings = PriorEmbedding(knowledges['promoter'].size(1), config.hidden_size)
                num_hidden+=1
            else:
                raise
        
        if self.use_co_exp:
            if knowledges.get('co_exp', None) is not None:
                self.register_buffer('co_exp_knowledge', knowledges['co_exp'])
                self.homologous_gene_human2mouse = knowledges['homologous_gene_human2mouse']
                token_num = knowledges['co_exp'].size(0)
                self.co_exp_embeddings = PriorEmbedding(knowledges['co_exp'].size(1), config.hidden_size)
                num_hidden+=1
            else:
                raise
        
        if self.use_gene_family:
            if knowledges.get('gene_family', None) is not None:
                self.register_buffer('gene_family_knowledge', knowledges['gene_family'])
                self.homologous_gene_human2mouse = knowledges['homologous_gene_human2mouse']
                token_num = knowledges['gene_family'].size(0)
                self.gene_family_embeddings = PriorEmbedding(knowledges['gene_family'].size(1), config.hidden_size)
                num_hidden+=1
            else:
                raise
        
        if self.use_peca_grn:
            if knowledges.get('peca_grn', None) is not None:
                self.register_buffer('peca_grn_knowledge', knowledges['peca_grn'])
                self.homologous_gene_human2mouse = knowledges['homologous_gene_human2mouse']
                token_num = knowledges['peca_grn'].size(0)
                self.peca_grn_embeddings = PriorEmbedding(knowledges['peca_grn'].size(1), config.hidden_size)
                num_hidden+=1
            else:
                raise
            
        if self.use_values:
            if self.concat_values_first:
                print("directly concat value")
                self.values_embeddings = nn.Sequential(OrderedDict([
                        ("c_fc", nn.Linear(config.hidden_size + self.value_dim, config.hidden_size)),
                        ("c_ln", nn.LayerNorm(config.hidden_size)),
                        ("gelu", QuickGELU()),
                        ("c_proj", nn.Linear(config.hidden_size, config.hidden_size))
                    ]))
                self.concat_embeddings = nn.Sequential(OrderedDict([
                        ("cat_fc", nn.Linear(config.hidden_size*(1+num_hidden), config.hidden_size)),
                        ("cat_ln", nn.LayerNorm(config.hidden_size)),
                        ("cat_gelu", QuickGELU()),
                        ("cat_proj", nn.Linear(config.hidden_size, config.hidden_size))
                    ]))
            else:
                self.concat_embeddings = nn.Sequential(OrderedDict([
                        ("cat_fc", nn.Linear(config.hidden_size*(1+num_hidden)+self.value_dim, config.hidden_size)),
                        ("cat_ln", nn.LayerNorm(config.hidden_size)),
                        ("cat_gelu", QuickGELU()),
                        ("cat_proj", nn.Linear(config.hidden_size, config.hidden_size))
                    ]))

        else:
            self.concat_embeddings = nn.Sequential(OrderedDict([
                        ("cat_fc", nn.Linear(config.hidden_size*(1+num_hidden), config.hidden_size)),
                        ("cat_ln", nn.LayerNorm(config.hidden_size)),
                        ("cat_gelu", QuickGELU()),
                        ("cat_proj", nn.Linear(config.hidden_size, config.hidden_size))
                    ]))                
            #  
        # 以下是原始 BertEmbedding 部分（除去word_embeddings）

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        if self.use_promoter or self.use_co_exp or self.use_gene_family or self.use_peca_grn:
            homologous_index = torch.arange(token_num)
            homologous_index[list(self.homologous_gene_human2mouse.keys())] = torch.as_tensor(list(self.homologous_gene_human2mouse.values()))
        self.register_buffer('homologous_index', homologous_index)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        values: Optional[torch.FloatTensor] = None,
        species: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        emb_warmup_alpha: float = 1.,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # value concat
        if self.use_value_emb:
            values = self.values_embeddings(values)
        else:
            values = values.unsqueeze(-1).float()
        if self.use_values:
            if self.concat_values_first:
                inputs_embeds = self.values_embeddings(torch.cat([inputs_embeds, values], dim=2))
            else:
                inputs_embeds = torch.cat([inputs_embeds, values], dim=2)

        input_ids_shifted = input_ids.detach().clone()

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        species = torch.ones(input_shape[0], 1).to(device).long()
        
        if species is not None:
            input_ids_shifted[species.squeeze(1) == 1] = self.homologous_index[input_ids_shifted[species.squeeze(1) == 1]]

        if self.use_promoter:
            promoter_inputs = self.promoter_knowledge[input_ids_shifted]
            promoter_embeds = self.promoter_embeddings(promoter_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * promoter_embeds), dim=2)
        if self.use_co_exp:
            co_exp_inputs = self.co_exp_knowledge[input_ids_shifted]
            co_exp_embeds = self.co_exp_embeddings(co_exp_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * co_exp_embeds), dim=2)
        if self.use_gene_family:
            gene_family_inputs = self.gene_family_knowledge[input_ids_shifted]
            gene_family_embeds = self.gene_family_embeddings(gene_family_inputs)
            inputs_embeds = torch.cat((inputs_embeds, emb_warmup_alpha * gene_family_embeds), dim=2)
        if self.use_peca_grn:
            peca_grn_inputs = self.peca_grn_knowledge[input_ids_shifted]
            peca_grn_embeds = self.peca_grn_embeddings(peca_grn_inputs)
            inputs_embeds = torch.cat((inputs_embeds,emb_warmup_alpha * peca_grn_embeds), dim=2)
        
        inputs_embeds = self.concat_embeddings(inputs_embeds)
        
        # 以下是原始 BertEmbedding 部分
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings