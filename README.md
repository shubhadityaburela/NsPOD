## AUTOMATED TRANSPORT SEPARATION USING THE NEURAL SHIFTED PROPER ORTHOGONAL DECOMPOSITION (Base version)

This repo contains the source code and jupyter notebooks for an automated transport separation and model reduction framework with the help of neural networks. The script (`.ipynb`) files for all the test cases are present here.
* `Crossing_waves.ipynb` describes the application and results for the application of our method to a synthetically generated crossing wave data set.
* `Wildfire.ipynb` performs the transport separation and model reduction for a 1D wildland fire model. The snapshot data are also provided for this example in the folder `Wildfire_input`.

The reader is encouraged to try out the examples on their own. We have however, provided the already trained weights for both the examples in the form `Crossing_waves.pth` and `Wildfire_alreadyTrained.pth`. 

Note: This is not the most up-to-date repository. Kindly refer [here](https://github.com/MOR-transport/automated_NsPOD) for the latest version and future developments.
