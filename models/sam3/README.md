# Local SAM model store

`detector.checkpoint` in `config.yaml` points at a file here (path relative to `config.yaml`). No
Hugging Face access is needed at run time. Simulation run configs are written with the absolute path.

| File | Format | Used |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | SAM 3.1, original sam3 naming (`detector.*`, `tracker.*`), fp16 | **yes** (configured) |
| `sam3.safetensors` | SAM 3, transformers naming (`detector_model.*`) | no: the `sam3` package can't load it |

Also accepted: the original PyTorch files `sam3.pt` (facebook/sam3) and `sam3.1_multiplex.pt`
(facebook/sam3.1).

SAM 3.1's checkpoint is built for video tracking. For photos and sampled video frames, the detector
loads only its text-prompted detector part (`detector.*`) and runs it on single images; the tracker
weights wait for the video phase. fp16 weights are loaded into the fp32 model.

The loader checks that every weight of the chosen model is filled, and stops with an error otherwise,
instead of running an untrained model.

Check: `.venv\Scripts\python scripts\check_sam3.py sample-files\kolkata_ultadanga_junction.jpg`
(the log line "SAM 3.1 detector on cuda:0" confirms which model was loaded).
