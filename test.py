#%%
import argparse

parser = argparse.ArgumentParser(description="Data from APL requests")
parser.add_argument("--object", "-o", type=str, default="merc")
parser.add_argument("--center-yr", '-cy', type=int, default=1870)
parser.add_argument("--length", '-l', type=int, default=40)
args = parser.parse_args()
print(args)

