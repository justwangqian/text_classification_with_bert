import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class NeuralNet(nn.Module):
    def __init__(self, model_name, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained(model_name, num_labels=num_class)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.config.hidden_size * 2, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):
        output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        last_hidden = output.last_hidden_state
        all_hidden_states = output.hidden_states
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        f = torch.mean(last_hidden, 1)
        feature = torch.cat((feature, f), 1)
        h = torch.zeros()

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)





