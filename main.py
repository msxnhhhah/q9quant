import subprocess
import os
import sys

# Get the path of the current Python interpreter
interpreter = sys.executable

# Define the folder containing the scripts
folder = 'real_time'

# List of scripts to run
scripts = ['binance_btc_futures_real_time.py', 'okx_btc_futures_real_time.py', 'binance_btc_spot_real_time.py', 'full_backtesting_real_time.py', 'ic_ir_calculation.py']

for script in scripts:
    # Create the full path to the script
    script_path = os.path.join(folder, script)
    
    if os.path.exists(script_path):
        print(f"Running {script_path}...")
        # Run the script using the current Python interpreter
        result = subprocess.run([interpreter, script_path])
        if result.returncode == 0:
            print(f"{script_path} completed successfully")
        else:
            print(f"{script_path} failed with return code {result.returncode}")
    else:
        print(f"Script {script_path} does not exist")