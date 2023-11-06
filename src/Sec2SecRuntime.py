# %%
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.linear(outputs)
        return predictions, hidden, cell


# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_steps_out):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.n_steps_out = n_steps_out

    def forward(self, x):
        hidden, cell = self.encoder(x)
        decoder_input = torch.zeros(
            (x.size(0), self.n_steps_out, self.decoder.lstm.input_size)
        ).to(x.device)
        outputs, _, _ = self.decoder(decoder_input, hidden, cell)
        return outputs


# %%
class Seq2SeqRuntime:
    def __init__(self, name: str):
        # Load the checkpoint

        device = (
            torch.device("CUDA") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.checkpoint = torch.load(name, map_location=torch.device(device))
        # Extract the scaler from the checkpoint
        self.scaler = self.checkpoint["scaler"]

    def preprocess_data_for_inference(self, df, n_steps_in, n_steps_out):
        # Normalize the data
        data_normalized = self.scaler.transform(
            df
        )  # Note: use transform, not fit_transform
        # print(data_normalized.shape)
        X = self.create_sequences_for_inference(
            data_normalized, n_steps_in, n_steps_out
        )
        # print(X.shape)
        return torch.tensor(X, dtype=torch.float32)

    def create_sequences_for_inference(self, data, n_steps_in, n_steps_out):
        X = []
        for i in range(0, len(data) - n_steps_in - n_steps_out + 5):
            seq_x = data[i : i + n_steps_in]
            if seq_x.shape[0] == n_steps_in:
                X.append(seq_x)
        return np.array(X)

    # this function needs to do all the work instead of `create_sequences_for_inference`
    def process_runtime_data(self, df):
        data_normalized = self.scaler.transform(df)
        X = np.array(data_normalized)
        return torch.tensor(X, dtype=torch.float32)

    def predict(self, eval_df):
        """
        Safety-Critical Applications: If the prediction is used for real-time safety systems
        (like advanced driver-assistance systems, ADAS), the most recent predictions may be the most
        valuable as they can inform immediate safety interventions.

        Driver Profiling or Long-Term Trends: If the goal is to understand long-term driver behavior
        for insurance purposes or driver coaching, then averaging or aggregating over a range of predictions
        to get a more stable and generalized profile might be more appropriate.


        """

        # print(f"X_input_tensor", eval_df.shape)
        # Preprocess the data
        X_new_tensor = self.preprocess_data_for_inference(eval_df, 10, 0)
        # X_new_tensor = process_runtime_data(eval_df) # Need to use something like this instead
        # print(f"X_new_tensor", X_new_tensor.shape)
        # Initialize the model based on the shape of the input data
        model = Seq2Seq(
            input_size=X_new_tensor.shape[2],
            hidden_size=50,
            n_steps_out=10,
            output_size=X_new_tensor.shape[2],
        )

        # Load model's state dictionary from the checkpoint
        model.load_state_dict(self.checkpoint["model_state_dict"])

        # Predict using the model
        model.eval()
        with torch.no_grad():
            y_new_pred_tensor = model(X_new_tensor)
            y_new_pred = y_new_pred_tensor.numpy()
        # print(f"y_new_pred_tensor",y_test_tensor.shape)
        # print(y_new_pred.shape)

        # Take the first prediction from the first sequence
        y_first_pred = y_new_pred[0, :, :]
        # print(f"first predict", y_first_pred.shape)
        # print(y_first_pred)
        # Inverse transform the predictions to the original scale
        y_first_pred_original = self.scaler.inverse_transform(y_first_pred)

        return y_first_pred_original

        # Extract the denormalized delta_velocity values
        # delta_velocity_pred_original = y_first_pred_original[:, delta_velocity_index]

        # Return the predicted delta velocity
        # return delta_velocity_pred_original


# %% using this model
# import pandas as pd

# # sample data

# model = Seq2SeqRuntime("model_scaler_cluster_1.pth")

# runtime_data = pd.DataFrame(
#     {
#         "delta_position": [
#             25.147708,
#             24.986054,
#             24.847105,
#             24.691872,
#             24.533773,
#             24.374004,
#             24.209939,
#             24.042500,
#             23.879786,
#             23.715263,
#         ],
#         "delta_velocity": [
#             1.614755,
#             1.589909,
#             1.589107,
#             1.587685,
#             1.591374,
#             1.592214,
#             1.621730,
#             1.656232,
#             1.668315,
#             1.687577,
#         ],
#         "v_follower": [
#             11.265286,
#             11.227821,
#             11.216660,
#             11.199787,
#             11.174766,
#             11.139135,
#             11.118996,
#             11.104114,
#             11.075143,
#             11.052290,
#         ],
#     }
# )

# # Sample usage
# delta_velocity_pred = model.predict_delta_velocity(runtime_data, 1)
# print(f"Predicted result:", delta_velocity_pred)
