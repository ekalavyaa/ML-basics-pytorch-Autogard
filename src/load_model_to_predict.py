import torch
import argparse as arg

model = torch.load('model/model.pt')

parser = arg.ArgumentParser(description='input value to predict')

parser.add_argument('--foo', type=int, default=42, help='FOO!')

args = parser.parse_args()
x_input = torch.tensor([[args.foo]], dtype = torch.float)

y_pred = model(x_input)

print(y_pred)