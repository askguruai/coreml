
To run the service, you have to set env
```
export OPENAI_API_KEY='sk-...'
```

# ML Core Service

ML Core Service is able to get embeddings from text(s) and generate response based on query and text(s).


## API

- Interactive documentation available in [swagger](http://78.141.213.164:5555/docs).

### POST /embeddings

<details>
  <summary>Creates an embedding vector representing the input text.</summary>
  

  | Parameter |         Type         |                                                                                          Description                                                                                          | Optional |
  |:---------:|:--------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|
  |  `input`  | `str` or `List[str]` | Input text to get embeddings for, encoded as a string. To get embeddings for multiple inputs in a single request, pass an array of strings. Each input must not exceed 8192 tokens in length. | False    |


#### Examples

<details>
  <summary>Create an embedding for single text.</summary>
  
  ```bash
  curl -X 'POST' \
    'http://78.141.213.164:5555/embeddings/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "input": "vp rnd"
  }'
  ```
  
  Response:
  
  ```json
  {
    "data": [
      {
        "object": "embedding",
        "index": 0,
        "embedding": [
          -0.004258352797478437,
          -0.024816041812300682,
          ...
          0.0022093546576797962
        ]
      }
    ]
  }
  ```
</details>

  Response:
  
  ```
  {
    "data": [
      {
        "object": "embedding",
        "embedding": [
          0.0023064255,
          -0.009327292,
          .... (1056 floats total for ada)
          -0.0028842222,
        ],
        "index": 0
      }
    ],
  }
  ```
</details>




## Development

Clone repo
```bash
git clone git@github.com:askaye/coreml.git
cd ./coreml
```

Install dependencies
```
conda create --name ml python=3.10
conda activate ml
make install
pip install black isort
```

Run
```
python main.py
```

### extra

For logging use
```python
import logging
logging.info("sample log")
```

To use variables from config use
```python
from utils import CONFIG
sample_var = CONFIG["sample_key"]["sample_var"]
```

## Deployment

TODO
