# Commands to run

```bash
cd video-captioning-agent                            # move into the project folder

export INPUT_PATH="$(pwd)/sample_input/tasks.json"   # tell main.py where to read tasks from (instead of the default /input/tasks.json)
export OUTPUT_PATH="$(pwd)/sample_output_2/results.json" # tell main.py where to write results (instead of the default /output/results.json)
python3 main.py                                      # run the pipeline: caption every clip in tasks.json, in all requested styles
cat sample_output/results.json                       # print the generated captions
```

## Optional: detailed per-clip descriptions (not part of grading)

```bash
export DESCRIBE_OUTPUT_PATH="$(pwd)/sample_output/descriptions.json" # where to write the long descriptions
python3 describe_videos.py                                          # run the same clips through a richer "describe everything" prompt
cat sample_output/descriptions.json                                 # print the detailed descriptions
```

## Optional: LLM-as-judge scoring of a results.json (not part of grading)

```bash
export RESULTS_PATH="$(pwd)/sample_output/results.json"             # captions to grade (from main.py)
export JUDGE_OUTPUT_PATH="$(pwd)/sample_output/judged_results.json" # where to write scores
python3 judge.py                                                    # score every caption for accuracy + tone_fit
cat sample_output/judged_results.json
```

To change the submitted generator model, edit `CLAUDE_MODEL_ID` in `.env` (defaults to
`claude-sonnet-5`), then rerun. Fireworks settings are only needed for judge/dev
experiments.
