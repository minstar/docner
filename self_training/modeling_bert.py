from transformers import BertPreTrainedModel,BertForTokenClassification, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn.utils.rnn  import pack_padded_sequence

from torch.autograd import Variable
from crf_made import CRF
from torch.nn import CrossEntropyLoss, KLDivLoss

class BERTForTokenClassification_v2(BertForTokenClassification):

    base_model_prefix = "bert"
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size*2, config.num_labels)
        # self.crf = CRF(self.num_labels, batch_first=True)
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=2)
        self.lambda1 = 1e-1
        self.lambda2 = 1e-3
        self.epsilon = 1e-8
        self.threshold = 0.3

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, label_mask=None, entity_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        device = input_ids.device

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        """ Bilstm for label refinement """
        if entity_ids is not None:
            entity_ids = entity_ids[:,:,None]
            bilstm_hidden = self.rand_init_hidden(batch_size)
            
            fst_bilstm_hidden = bilstm_hidden[0].to(device)
            bst_bilstm_hidden = bilstm_hidden[1].to(device)

            lstm_out, lstm_hidden = self.bilstm(sequence_output, (fst_bilstm_hidden, bst_bilstm_hidden))
            lstm_out = lstm_out.contiguous().view(-1, self.config.hidden_size*2)
            d_lstm_out = self.dropout(lstm_out)
            l_out = self.classifier2(d_lstm_out)
            lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)

            """ make representation similar on same or synonym entities (without regarding to context representation) """
            sft_logits = self.softmax(logits)
            sft_feats = self.softmax(lstm_feats)
            kl_logit_lstm = F.kl_div(sft_logits.log(), sft_feats, None, None, 'sum')
            kl_lstm_logit = F.kl_div(sft_feats.log(), sft_logits, None, None, 'sum')
            kl_distill = (kl_logit_lstm + kl_lstm_logit) / 2

            """ update entities with lstm and mlp classifier """
            lstm_feats = lstm_feats * entity_ids # mask for only updated entities

            """ update through uncertainties """
            uncertainty = -torch.sum(logits * torch.log(logits + self.epsilon), dim=2)
            ones = torch.ones(uncertainty.shape).to(device)
            zeros = torch.zeros(uncertainty.shape).to(device)

            uncertainty_mask = torch.where(uncertainty > self.threshold, ones, zeros)
            lstm_uncertainty_mask = uncertainty_mask[:,:,None]
            lstm_feats = lstm_feats * lstm_uncertainty_mask

        outputs = (logits,sequence_output) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]

            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if entity_ids is not None:
                active_lstm_logits = lstm_feats.view(-1, self.num_labels)[active_loss]
                lstm_loss = loss_fct(active_lstm_logits, active_labels)
                final_loss = loss + (self.lambda1) * lstm_loss + (self.lambda2) * kl_distill
                # lstm_crf = self.crf(lstm_feats, labels, mask=attention_mask)
                # final_loss = loss + (-self.lambda1) * lstm_crf + (self.lambda2) * kl_distill
                outputs = (final_loss,) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)

    def rand_init_hidden(self, batch_size,):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * 2, batch_size, self.config.hidden_size)), Variable(torch.randn(2 * 2, batch_size, self.config.hidden_size))
