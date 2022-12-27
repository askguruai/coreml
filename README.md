# ML Core Service

ML Core Service is able to get embeddings from text(s) and generate response based on query and text(s).


## API

- Interactive documentation available in [swagger](http://78.141.213.164:5555/docs).

### POST /embeddings

Creates an embedding vector representing the input text.

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

<details>
<summary>Create embeddings for multiple texts.</summary>

```bash
curl -X 'POST' \
  'http://78.141.213.164:5555/embeddings/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": ["terkom ceo", "Андрей Николаевич Терехов — доктор физико-математических наук, профессор, заведующий кафедрой СП СПбГУ."]
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
        0.009389051236212254
      ]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [
        -0.021062027662992477,
        0.014139993116259575,
        ...
        0.0022093546576797962
      ]
    }
  ]
}
```
</details>

### POST /completions

Answers given question based on provided infromation.

| Parameter |  Type |                  Description                  | Optional |
|:---------:|:-----:|:---------------------------------------------:|:--------:|
|  `query`  | `str` |        Question to provide answer for.        |   False  |
|   `info`  | `str` | Context for the question. Just chunk of text. |   False  |

#### Examples

<details>
<summary>Get an answer for a question</summary>

```bash
curl -X 'POST' \
  'http://78.141.213.164:5555/completions/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Why it happened?",
  "info": "On 24 February 2022, Russia invaded Ukraine in a major escalation of the Russo-Ukrainian War, which began in 2014. The invasion has resulted in tens of thousands of deaths on both sides. It has caused Europe'\''s largest refugee crisis since World War II.[10][11] An estimated 8 million Ukrainians were displaced within their country by late May and 7.8 million fled the country by 8 November 2022,[12][13][14][15] while Russia, within five weeks of the invasion, experienced its greatest emigration since the 1917 October Revolution.[16]\r\n\r\nFollowing the 2014 Ukrainian Revolution, Russia annexed Crimea, and Russian-backed paramilitaries seized part of the Donbas region of south-eastern Ukraine, which consists of Luhansk and Donetsk oblasts, sparking a regional war.[17][18] In March 2021, Russia began a large military build-up along its border with Ukraine, eventually amassing up to 190,000 troops and their equipment. Despite the build-up, denials of plans to invade or attack Ukraine were issued by various Russian government officials up to the day before the invasion.[22] On 21 February 2022, Russia recognised the Donetsk People'\''s Republic and the Luhansk People'\''s Republic, two self-proclaimed breakaway quasi-states in the Donbas.[23] The next day, the Federation Council of Russia authorised the use of military force and Russian troops entered both territories.[24]\r\n\r\nThe invasion began on the morning of 24 February 2022,[25] when Russian president Vladimir Putin announced a \"special military operation\"[26] aiming for the \"demilitarisation\" and \"denazification\" of Ukraine.[27][28] In his address, Putin espoused irredentist views,[29] challenged Ukraine'\''s right to statehood,[30][31] and falsely[32] claimed Ukraine was governed by neo-Nazis who persecuted the ethnic Russian minority.[33] Minutes later, Russian strikes and a large ground invasion were launched on a northern front from Belarus towards Kyiv, a north-eastern front towards Kharkiv, a southern front from Crimea, and a south-eastern front from Luhansk and Donetsk.[34][35][36] Ukrainian president Volodymyr Zelenskyy enacted martial law and a general mobilisation.[37][38] Russian troops retreated from the northern front by April. On the southern and south-eastern fronts, Russia captured Kherson in March and then Mariupol in May after a siege. On 18 April, Russia launched a renewed attack on the Donbas region. Russian forces continued to bomb both military and civilian targets far from the frontline, including electrical and water systems.[39][40][41] In late 2022, Ukrainian forces launched counteroffensives in the south and in the east. Soon after, Russia announced the illegal annexation of four partially occupied oblasts. In November, Ukraine retook the city of Kherson.\r\n\r\nThe invasion has received widespread international condemnation. The United Nations General Assembly passed a resolution condemning the invasion and demanding a full withdrawal of Russian forces.[42] The International Court of Justice ordered Russia to suspend military operations and the Council of Europe expelled Russia. Many countries imposed sanctions on Russia, as well as on its ally Belarus, which have affected the economies of Russia and the world,[43] and provided humanitarian and military aid to Ukraine,[44] totaling over $80 billion from 40 countries as of August 2022.[45] Protests occurred around the world; those in Russia were met with mass arrests and increased media censorship,[46][47] including a ban on the words \"war\" and \"invasion\".[48][49] Over 1,000 companies have pulled out of Russia and Belarus in response to the invasion.[50] The International Criminal Court has opened an investigation into crimes against humanity in Ukraine since 2013, including war crimes in the 2022 invasion.[51]"
}'
```

Response:
```json
{
  "data": " The invasion began on the morning of 24 February 2022, when Russian president Vladimir Putin announced a \"special military operation\" aiming for the \"demilitarisation\" and \"denazification\" of Ukraine. In his address, Putin espoused irredentist views, challenged Ukraine's right to statehood, and falsely claimed Ukraine was governed by neo-Nazis who persecuted the ethnic Russian minority."
}
```
</details>


## Development

1. Clone repo
  ```bash
  git clone git@github.com:askaye/coreml.git
  cd ./coreml
  ```

2. Install dependencies
  ```bash
  conda create --name ml python=3.10
  conda activate ml
  make install
  pip install black isort
  ```

3. Set environment variable
  ```bash
  export OPENAI_API_KEY='sk-...'
  ```

4. Run service
  ```bash
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

## ToDo

- add text completion endpoint
- add deployment code using docker
    * add launch guides to readme
