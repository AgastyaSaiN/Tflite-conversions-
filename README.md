# Base pt conversions and testing

Put a YOLOv5 `.pt` file in, get TFLite variants out, with a benchmark against the model's own FP32 baseline.

```
abcd123.pt  -->  abcd123_fp32.tflite          baseline (same size, no quantization)
                 abcd123_m1_pc / _pt.tflite   M1: everything INT8
                 abcd123_m2_pc / _pt.tflite   M2: backbone INT8, neck and head float
                 abcd123_m3_pc / _pt.tflite   M3: backbone and neck INT8, head float
```

`pc` = per-channel quantization, `pt` = per-tensor quantization.
The image size inside the `.pt` is kept (640 stays 640).

## Quick start

1. Copy the `.pt` file into **`put_pt_files_here`**.
2. Optional, for accuracy numbers: put labelled test pictures into **`put_test_pictures_here`** (see the README.txt in that folder).
3. Optional, better INT8 quality: put pictures like your real ones into **`put_calibration_pictures_here`**.
4. Run the check, then the factory:

| | Windows | Linux |
|---|---|---|
| check | double-click `CHECK_SETUP.bat` | `./CHECK_SETUP.sh` |
| run | double-click `RUN.bat` | `./RUN.sh` |

5. Open `results/index.html`. Each model has its own folder in `results/<model name>/`:
   - `models/` holds the `.tflite` files.
   - the report is an HTML file with the size, speed, memory and accuracy of every variant against the FP32 baseline; `results.csv` holds the same numbers.
   - if something failed, `ERROR.txt` explains it in plain words.

## Things to know

- **Only YOLOv5 `.pt` files** (trained with the YOLOv5 repository). YOLOv8 and YOLO11 files are not accepted.
- **Time.** Each INT8 model takes several minutes to build (about 7 minutes each on a laptop). A full conversion of one model takes roughly 30 to 50 minutes, the benchmark about 10 to 20 more.
- **Interrupting is safe.** Start the run again and finished steps and finished model files are skipped. `python run.py --force` redoes everything.
- **Without test pictures** you still get the model files and size, speed and memory, but no accuracy.
- **Calibration pictures** should look like the real deployment pictures. Poor calibration pictures give a weaker INT8 model.
- **INT8 loses some accuracy.** The report shows how much each variant keeps compared with the FP32 baseline (column "retained"). Judge M1, M2 and M3 by that number.
- **Speed and memory** are measured on the computer that runs the tool, with 4 threads on separate physical cores, and are only comparable on the same computer. A phone or NPU will behave differently.
- **The box numbers are rescaled to 0-1 inside the model** so the all-INT8 model (M1) works. The tool undoes this automatically when it measures accuracy.

## Options

```
python run.py              normal run
python run.py --quick      short trial: 100 calibration pictures, 200 test pictures, 3 speed rounds
python run.py --force      redo everything
python run.py --skip-speed skip the speed and memory measurements
```

## If something is missing (offline PC)

`CHECK_SETUP` lists every missing package and prints the install command, for example:

```
pip install --no-index --find-links linux_offline_packages/wheelhouse "tensorflow==2.15.1"
```

The tool looks for a Python that has everything. If the packages are split over two environments, that works too. To point it at a specific Python, set the environment variable `BASEPT_PYTHON` (or `BASEPT_PYTHON_EXPORT`, `BASEPT_PYTHON_TF`, `BASEPT_PYTHON_BENCH` for one job each).

### Linux x86_64, Python 3.11 (fully offline)

The folder `linux_offline_packages` holds the Miniconda installer, all Python packages and a checksum list. It needs glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL or Rocky 8+). Copy the whole folder, then:

```
bash SETUP_LINUX.sh      one time: checks the files, installs conda if needed, creates the environment "basept"
source activate_env.sh   activates it in a new terminal
./CHECK_SETUP.sh
./RUN.sh
```

`linux_offline_packages` is not stored in git because of its size. Copy it with a USB drive or network folder.

## What the variants mean

- **FP32 baseline**: your model converted to TFLite without quantization. All comparisons are against it.
- **M1**: every layer INT8. Smallest and fastest, largest accuracy loss.
- **M2**: backbone INT8, neck and head stay float.
- **M3**: backbone and neck INT8, only the head stays float. Usually the best balance.

## Folder map

```
put_pt_files_here/             input: .pt files
put_test_pictures_here/        input: labelled test pictures (optional)
put_calibration_pictures_here/ input: calibration pictures (optional)
results/                       output: models, reports
lib/                           the code of the four steps
vendor_yolov5/                 bundled YOLOv5 code used for the export
linux_offline_packages/        offline Python packages for Linux
```

The four steps: 1 export the `.pt` to ONNX, 2 convert to TFLite and quantize, 3 benchmark, 4 report.
