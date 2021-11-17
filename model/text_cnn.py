from transformers import AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F


class Text_CNN_Bert(nn.Module):
    def __init__(self,
                 model_name,
                 n_filters=128,
                 filter_sz=None,
                 classes=2):
        super(Text_CNN_Bert, self).__init__()
        if filter_sz is None:
            filter_sz = [2, 3, 4]
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, self.config.hidden_size))
             for filter_size in filter_sz]
        )
        self.fc = nn.Linear(len(filter_sz) * n_filters, classes)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):

        outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        # [batch_size, seq_len, config.hidden_size]
        last_hidden_state = outputs.last_hidden_state
        # [batch_size, 1, seq_len, config.hidden_size]
        last_hidden_state = last_hidden_state.unsqueeze(1)
        # [batch_size, num_filters, seq_len - filter_size+1]
        conved = [F.relu(conv(last_hidden_state)).squeeze(3) for conv in self.convs]
        # max over the time pooling
        # [batch_size, num_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat(pooled, dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(cat))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(cat))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)

