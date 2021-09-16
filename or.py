from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df
x1	x2	y
0	0	0	0
1	0	1	1
2	1	0	1
3	1	1	1
X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_OR = Perceptron(eta=ETA, epochs=EPOCHS)
model_OR.fit(X, y)

_ = model_OR.total_loss()