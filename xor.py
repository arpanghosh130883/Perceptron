from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

XOR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,0],
}

df = pd.DataFrame(XOR)

df
x1	x2	y
0	0	0	0
1	0	1	1
2	1	0	1
3	1	1	0
X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_XOR = Perceptron(eta=ETA, epochs=EPOCHS)
model_XOR.fit(X, y)

_ = model_XOR.total_loss()