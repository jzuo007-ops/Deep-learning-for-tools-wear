import scipy.io
import numpy as np

data = scipy.io.loadmat('3. Milling/mill.mat')
print(data.keys())
mill = data['mill']
print(mill.dtype)
print(len(mill[0]))

# Check one case
case = mill[0][0]
print(case.dtype.names)
print("Case ID:", case['case'][0][0])
print("Run ID:", case['run'][0][0])
print("VB:", case['VB'][0][0])
# print("Data shape:", case['smcAC'].shape) # This might be where the signal data is

# Let's see what's in smcAC for example
print("smcAC type:", type(case['smcAC']))
print("smcAC shape:", case['smcAC'].shape)
print("smcAC first element shape:", case['smcAC'][0].shape)
if len(case['smcAC'][0]) > 0:
    print("smcAC first element actual data shape:", case['smcAC'][0][0].shape)

# Let's check the length of signals for a few runs
for i in range(5):
    c = mill[0][i]
    # In MATLAB files loaded by scipy, structured arrays can be tricky.
    # Let's try to access the signals properly.
    # The fields are: 'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle'
    sig = c['smcAC']
    print(f"Run {i+1}: VB={c['VB']}, sig shape={sig.shape}")
    if sig.size > 0:
        # Accessing the actual signal vector
        actual_sig = sig.flatten()
        print(f"  Signal length: {len(actual_sig)}")
