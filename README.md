# ThinMNIST
Self implementation of ThinMNIST (in a DataLoader for PyTorch).

I strongly recommend using the [preprocessed version](https://drive.google.com/file/d/1zegZFCoZ50OVOrn3j7dgvw2y6Kwn_yTz/view?usp=sharing) with the `thin_mnist.py` DataLoader, instead of using `thin_mnist_slow.py`.

NOTE that images are padded with 2 pixels on each border, changin the MNIST image size from `28x28` to `32x32`!

**Thin:**
![](example_thin.png)

**Original:**
![](example_original.png)

Check `test.ipynb` for some output examples.
