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

To change the model, edit `RITS_VISION_API_ENDPOINT` / `RITS_VISION_MODEL_ID` in `.env`, then rerun.
