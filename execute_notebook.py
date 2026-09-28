import nbformat
from nbclient import NotebookClient
import time

print("Executing notebook top-to-bottom...")
start_time = time.time()

with open('DS_Day01_05_Banking_Customer_Transaction_Behaviour.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open('DS_Day01_05_Banking_Customer_Transaction_Behaviour.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook successfully executed and updated with outputs in {time.time() - start_time:.2f}s!")
