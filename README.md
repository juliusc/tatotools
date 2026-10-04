Installation

```
conda create -n tatotools --file environment.yml
```

Launching

```
conda activate tatotools
GT_API_KEY=<your api key>
python3 -m flask run --host=localhost --port=8000
```

Then navigate to `localhost:8000` in your browser.