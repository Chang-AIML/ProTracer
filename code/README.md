# ProTracer: Proprioception-Guided Failure Diagnosis in Robot Manipulation

Minimal implementation of ProTracer, with the FailTime benchmark annotations.

## Setup

```bash
pip install -e .
export OPENROUTER_API_KEY=...
```

FailTime-Long also needs `ffmpeg` with AV1 support.

## Data

```bash
ln -s /path/to/ViFailback/raw_data data/failtime_short/episodes
ln -s /path/to/FailTime-Long data/failtime_long/episodes
```

## Usage

```bash
python -m protracer diagnose data/failtime_short/test.json --out runs/short
python -m protracer evaluate runs/short data/failtime_short/test.json

# with the reflective experience from the paper
python -m protracer diagnose data/failtime_short/test.json --out runs/short_exp \
    --experience experience/gemini-3.1-pro.txt

# learn new experience
python -m protracer learn data/failtime_short/calibration.json --out experience/mine.txt
```

FailTime-Long runs the same way with `data/failtime_long/test.json`.
`--model` takes any OpenRouter model id.
