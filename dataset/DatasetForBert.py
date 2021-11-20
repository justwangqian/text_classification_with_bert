from torch.utils.data import Dataset
import torch


class CompDataset(Dataset):
    """
    A torch dataset reading data in dataframe
    """

    def __init__(self, df, tokenizer, MAX_LEN):
        self.tokenizer = tokenizer
        self.df_data = df
        self.length = df.shape[0]
        self.MAX_LEN = MAX_LEN

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.iloc[index, 0]
        sentence2 = self.df_data.iloc[index, 1]

        encoded_dict = self.tokenizer(sentence1,
                                      sentence2,
                                      max_length=self.MAX_LEN,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt'
                                      )

        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        # Convert the target to a torch tensor
        target = torch.tensor(self.df_data.iloc[index, 2])

        sample = (padded_token_list, att_mask, token_type_ids, target)

        return sample

    def __len__(self):
        return self.length


class TestDataset(Dataset):

    def __init__(self, df, tokenizer, MAX_LEN):
        self.tokenizer = tokenizer
        self.df_data = df
        self.length = df.shape[0]
        self.MAX_LEN = MAX_LEN

    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence1 = self.df_data.iloc[index, 0]
        sentence2 = self.df_data.iloc[index, 1]

        encoded_dict = self.tokenizer(sentence1,
                                      sentence2,
                                      max_length=self.MAX_LEN,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt'
                                      )

        # These are torch tensors already.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        sample = (padded_token_list, att_mask, token_type_ids)

        return sample

    def __len__(self):
        return self.length
