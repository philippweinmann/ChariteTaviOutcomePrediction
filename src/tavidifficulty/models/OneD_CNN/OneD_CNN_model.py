# %%
import torch

class BASIC_CNN1D(torch.nn.Module):
    # increase metadata_features when we add more
    def __init__(self, num_classes=1, input_channels=8, linDropoutRate=0.4, convDropoutRate=0.1):
        super(BASIC_CNN1D, self).__init__()
        # Input shape: (batch_size, 8, 400)
        self.starting_filters_amt = 16

        self.conv1 = torch.nn.Conv1d(
            input_channels, self.starting_filters_amt, kernel_size=3, padding=1
        )  # (16, 400)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.starting_filters_amt)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(convDropoutRate)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2)  # (16, 200)
        self.conv2 = torch.nn.Conv1d(
            self.starting_filters_amt,
            self.starting_filters_amt * 2,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm2 = torch.nn.BatchNorm1d(self.starting_filters_amt * 2)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(convDropoutRate)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.adaptivemaxpool = torch.nn.AdaptiveMaxPool1d(output_size=1)  # (64, 1)
        self.flatten = torch.nn.Flatten()

        # now the linear layers
        self.flatten = torch.nn.Flatten()
        self.dropout_fc = torch.nn.Dropout(linDropoutRate)
        self.linear1 = torch.nn.Linear(self.starting_filters_amt * 2, num_classes)

    def forward(self, mrt_slice_input):
        mrt_slice_input = self.conv1(mrt_slice_input)
        mrt_slice_input = self.batch_norm1(mrt_slice_input)
        mrt_slice_input = self.relu1(mrt_slice_input)
        mrt_slice_input = self.dropout1(mrt_slice_input)
        mrt_slice_input = self.maxpool1(mrt_slice_input)
        mrt_slice_input = self.conv2(mrt_slice_input)
        mrt_slice_input = self.batch_norm2(mrt_slice_input)
        mrt_slice_input = self.relu2(mrt_slice_input)
        mrt_slice_input = self.dropout2(mrt_slice_input)
        mrt_slice_input = self.maxpool2(mrt_slice_input)
        mrt_slice_input = self.adaptivemaxpool(mrt_slice_input)
        mrt_slice_input = self.flatten(mrt_slice_input)

        mrt_slice_input = self.dropout_fc(mrt_slice_input)
        mrt_slice_input = self.linear1(mrt_slice_input)

        return mrt_slice_input