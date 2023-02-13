# 

This repository contains the end-of-semester project by Benedikt Bauer and David Heim for the "Praktikum Programming Parallel Embedded Systems" course at University of Augsburg in the winter-term 2022/23.

The goal of the project was to implement a basic neural network library to train networks for inference on embedded devices like the Raspberry Pi Pico. You can find an overview of the initial project plan [here](docs/PLAN.md).

## Getting started

The project is split into a library component containing all the neural-network functionality in the `nn` folder, unit tests in `tests` and applications building on the library in `src`. 

### Building on x86

Building on x86 is fairly straight-forward, cmake is used as a build-configuration tool.

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=<Debug/Release> ..
make
```

The default `make` target will build all applications in `src`. To build just a single one run `make <TARGET>`, e.g. `make mnist_train`.

The `Debug` configuration additionally builds with gcc's address and undefined-behaviour sanitizers enabled.

### Cross-compiling for the RPi Pico

All applications written for the Pico are in the `src/pico` directory.
To cross compile these on x86 for the Pico pass the `BUILD_PICO` flag to cmake:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PICO=1 ..
```

This will automatically clone the Pico SDK, including required subprojects like tinyusb.

To flash an application to the Pico run
```sh
make upload_<TARGET> # e.g. upload_mnist_pico
```

### Tests

To build the unit-tests make sure you have not configured cmake with the `BUILD_PICO` flag enabled as that will disable the test targets. Then it is as simple as calling
```sh
make tests
ctest
```
to build and run all unit tests.

## Documentation

More documentation for the different parts of the project can be found in the [`docs/`](docs/) folder.