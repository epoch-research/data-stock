This repository contains the code used in [Will we run out of data? Limits of LLM scaling based on human-generated data]()

Directory `v1` contains the code used in the October 2022 version of the paper.

## Running scripts

First buid and start the container with

```
docker build -t data-stocks .
docker run -d -v ./results:/usr/src/app/results --name data-stocks-cont data-stocks
```

Then, execute any script you want with

```
docker exec your-container-name python src/script_name.py
```

The plots will be saved in the `results` directory.
