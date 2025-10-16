# NsPOD

Small collection of example scripts demonstrating NsPOD approach.

## Files / Scripts
Each script can be run directly with Python from the repository root.

- **`Single_wave.ipynb`**  
  Runs the 1D traveling-wave example.

- **`Crossing_StraightCubic_waves.ipynb`**  
  Runs the 1D crossing-waves example (straight cubic amplitudes).

- **`Crossing_sine_StraightCubic_waves.ipynb`**  
  Runs the 1D crossing-waves example with sine / cosine amplitude modulation.

- **`1D_wildlandfire.ipynb`**  
  Runs the 1D wildland-fire example.

> Optimization and algorithm parameters are set inside each script for the corresponding example. Edit the scripts to change settings. The wildland fire data is available upon request.

## For running the GPU scripts
From the directory **`Script_files_for_GPU`**, run any example with:

```bash
python Single_wave.py
# or
python Crossing_StraightCubic_waves.py
# or
python 1d_wildlandfire.py
