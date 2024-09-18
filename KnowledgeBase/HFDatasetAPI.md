
# How to Use Huggingface API to search and parse huggingface hub Datasets

This guide demonstrates how to use specific functions to interact with the Huggingface API and retrieve dataset information.

## Prerequisites

Make sure you have the following libraries installed:

```bash
pip install requests pandas datasets huggingface_hub
```

### Function 1: `search_datasets`

This function allows you to search for datasets using a keyword and retrieve the top N datasets. 

```python
import requests

def search_datasets(keyword: str, top_n: int = 5, hf_token=None):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    search_url = "https://huggingface.co/api/datasets"
    params = {"search": keyword, "limit": top_n}
    response = requests.get(search_url, params=params, headers=headers)

    if response.status_code != 200:
        return f"Failed to fetch datasets. Status code: {response.status_code}"

    datasets = response.json()
    if not datasets:
        return f"No datasets found for the keyword: {keyword}"

    dataset_info = []
    for dataset in datasets:
        dataset_name = dataset['id']
        dataset_info.append({
            'Dataset Name': dataset_name,
            'Description': dataset.get('description', 'No description'),
        })

    return dataset_info

# Example Usage
datasets = search_datasets("text", top_n=3)
print(datasets)
```

### Function 2: `fetch_dataset_info`

This function fetches detailed metadata about a specific dataset, including size, splits, and columns.

```python
import requests

def fetch_dataset_info(dataset_name: str, hf_token=None):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    config_url = f"https://huggingface.co/api/datasets/{dataset_name}"
    response = requests.get(config_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch dataset configuration for {dataset_name}")

    config_data = response.json()
    size_mb = config_data.get('size', 0) / (1024 ** 2)  # Convert size to MB
    splits = config_data.get('splits', {})

    return {
        'Dataset Name': dataset_name,
        'Size (MB)': size_mb,
        'Splits': list(splits.keys())
    }

# Example Usage
dataset_info = fetch_dataset_info("imdb")
print(dataset_info)
```

### Function 3: `get_dataset_columns`

This function retrieves the column names of a dataset.

```python
from datasets import load_dataset

def get_dataset_columns(dataset_name: str):
    dataset = load_dataset(dataset_name, split='train')
    return list(dataset.column_names)

# Example Usage
columns = get_dataset_columns("imdb")
print(columns)
```

### Function 4: `get_sample_rows`

This function fetches a few sample rows from the dataset without loading the entire dataset.

```python
from datasets import load_dataset

def get_sample_rows(dataset_name: str, num_rows: int = 3):
    dataset = load_dataset(dataset_name, split='train', trust_remote_code="true")
    sample_rows = dataset.select(range(num_rows))
    return sample_rows.to_pandas().to_dict(orient="records")

# Example Usage
sample_rows = get_sample_rows("imdb")
print(sample_rows)
```

### Putting It All Together

You can combine the above functions to search for datasets and retrieve detailed metadata for the top N datasets.

```python
def get_detailed_dataset_info(keyword: str, top_n: int = 5):
    datasets = search_datasets(keyword, top_n)
    
    for dataset in datasets:
        dataset_name = dataset['Dataset Name']
        info = fetch_dataset_info(dataset_name)
        columns = get_dataset_columns(dataset_name)
        sample_rows = get_sample_rows(dataset_name)

        print(f"Dataset: {dataset_name}")
        print(f"Info: {info}")
        print(f"Columns: {columns}")
        print(f"Sample Rows: {sample_rows}")
        print("="*50)

# Example Usage
get_detailed_dataset_info("text", top_n=2)
```
