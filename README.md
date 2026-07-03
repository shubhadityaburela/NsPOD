# Tensor_Decomposition

Small collection of example scripts demonstrating tensor decomposition approach.

---

## Step 1: Install Docker

Docker provides the virtual environment that runs the code.

1. **Download Docker Desktop:** Go to the [Docker Desktop Official Website](https://www.docker.com/products/docker-desktop/) and download the installer for your operating system (Windows, Mac, or Linux).
2. **Install and Run:** Run the installer and follow the standard on-screen prompts.
3. **Crucial:** Once installation is complete, launch the **Docker Desktop** application and keep it running in the background.

---

## Step 2: Open Your Terminal & Project Directory

1. Unzip and place this repository onto your machine.
2. Open your command line interface:
   * **Windows:** Open **Command Prompt** or **PowerShell**.
   * **macOS / Linux:** Open **Terminal**.
3. Navigate (`cd`) into your extracted project folder. For example:
   ```bash
   cd path/to/your/extracted-folder/
   ```
   
## Step 3: Build the Docker environment

You fill find ```Dockerfile``` in that repo. Build the environment (might take a few minutes) with:
```bash
docker build -t nspod .
```


## Step 4: Run the docker environment 
Run the simulations and save outputs according to your machine. 
* Windows command prompt : 
```bash 
docker run --rm -v "%cd%:/app" nspod --run "Single_wave"
```
* Windows power shell : 
```bash 
docker run --rm -v  "${PWD}:/app" nspod --run "Single_wave"
```
* mac/Linux terminal: 
```bash 
docker run --rm -v "$(pwd):/app" nspod --run "Single_wave"
``` 
The options for arguments set after --run
* ```Single_wave```, 
* ```Crossing_StraightCubic_waves```, 
* ```Crossing_sine_StraightCubic_waves```, 
* ```Wildlandfire_1d```

> Optimization and algorithm parameters are set inside each script for the corresponding example. Edit the scripts to change settings. The wildland fire data is available upon request.


## Step 5: Saved files
* ```.npy``` files will be saved in the ```data/``` folder and ```.png``` files will be stored in ```plots/``` folder. 




## ⚠️ System Requirements & Platform Note

### "It Works on My Machine" (and on our compute cluster)

This codebase intermediately performs low-level Singular Value Decomposition (SVD) matrix operations. 

If you are trying to run the Docker container via **Docker Desktop on an Apple Silicon Mac (M1/M2/M3/M4)**, you will likely hit a low-level crash stemming from the linear algebra engine:
`** On entry to SLASCL parameter number 4 had an illegal value`

* **Why?** This is an active bug inside x86_64 virtualization/emulation layers (like QEMU) when translating single-precision AVX vector instructions to ARM64 architecture. 
* **The Reality:** Docker cannot magically make an emulated Intel Linux virtual machine look like physical Mac hardware. 

### How to actually run it:
1. **Natively on Host Machine:** Run the code natively on your machine inside the provided Conda environment. It executes on native macOS (using Apple MPS/CPU accelerators) and native Linux clusters (using Nvidia CUDA).
2. **Native ARM64 Container:** If you absolutely must use Docker on a Mac, build the image specifically targeting your architecture (Not recommended though):
   ```bash
   docker build --platform linux/arm64 -t nspod .
   docker run --rm --platform linux/arm64 -v "$(pwd):/app" nspod --run "Single_wave"
   ```
