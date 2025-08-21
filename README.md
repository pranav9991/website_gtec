Scenario: E-commerce clicks/purchases stored as time-ordered events.

Approach:

Encoder LSTM encodes user session events.

Decoder LSTM tries to reconstruct the original sequence of actions.

Useful for detecting fraudulent behavior if reconstruction fails.

Dataset (CSV):

Columns: user_id, timestamp, action_type, product_id.
