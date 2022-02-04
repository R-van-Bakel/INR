# INR
Implicit neural representations capable of upsampling image pixels.  
See `HD_INR_env.yml` for the required anaconda setup. The code was made and tested on a Windows 10 machine.

In this repository you will find multiple implementations for implicit neural representations (INRs). The INR folder contains the actual library.

## Classes
This repository contains three different types of classes. Firstly we have `INRBaseClass` which defines a wrapper base class.
This class adds a fitting method and provides the base setup for any INR. A subclass is expected to define a specific model.  
Secondly, we have `MLP` and `Gabor`. These classes define an MLP and Gabor INR and inherit from `INRBaseClass`.  
Finally, we have `MLModel` and `GaborModel`. These classes are used to create the models that the aforementioned INR classes use.

## Parameters
### Parameter files
Firstly, note that all model hyperparameters are stored in separate `.yaml` files.
While these parameters can directly be set in the code, it keeps the code clearer if theses parameters are set elsewhere.  
For the Siren a `.yaml` could look like:  
```
codomain: [3]
hidden_size: 13
no_layers: 5
activ_func: sine
final_non_linearity: sigmoid
omega0: 1.0
omega0_initial: 0.02
dim_linear: 2
```

For the Gabornet a `.yaml` could look like:  
```
net:
  codomain: [3]
  hidden_channels: 15
  no_layers: 5
  input_scale: 256.0
  weight_scale: 0.1
  alpha: 1.0 # 6.0
  beta: 100.0 # 1.0
  bias: True
  init_spatial_value: 1.0
  covariance: anisotropic
  final_non_linearity: sigmoid
  dim_linear: 2
train:
  lr: 0.01
regularize_params:
  res: 32 # 0
  factor: 1.00 # 0.001
  gauss_stddevs: 2.0 # 2.0
  gauss_factor: 0.01 # 0.5
  target: gabor
  method: summed
```

### Loading in parameters
These `.yaml` files can be loaded in using Omegaconf:
```
from omegaconf import OmegaConf
cfg_sine = OmegaConf.load('./configs/RELU_SINE_INR_cfg.yaml')
cfg_gabor = OmegaConf.load('./configs/Gabor_INR_cfg.yaml')
```

### Creating a model
When these configuration are loaded we can create a Siren and Gabornet as follows:
```
sine_MLP_INR = MLP_INRClass(**cfg_sine)
Gabor = Gabor_INRClass(**cfg_gabor.net, **cfg_gabor.regularize_params)
```

## Training
In order to train these models we will need to provide some arguments. Firstly, we need to provide a set of coordinates to train on.
We have provided two helper functions for creating the coordinate grids.

### Creating coordinate grids
The `coordinate_grid` function takes in ranges and sizes in order to create a cartesian product of linspaces.
We could for example create a coordinate grid as follows:
```
domain = [[0,31][0,31]]
size = [100, 100]
some_grid = coordinate_grid(domain, size, reshape=False)
```
This would yield a tensor with size `[10000, 2]`. Each of the 2 dimensional entries in this tensor is a coordinate within the specified domain.
In this case we have 10,000 coordinates, because we have taken 100 samples of dimension 1 and 100 samples of dimension 2.
All coordinates would lie between `[0.,0.]` and `[31.,31.]` in this example.  
Note that there is also a `reshape` parameter. If we set this to `True` the batch dimension will be spit up according to the size parameter.
In the previous example this would yield a tensor with size `[100, 100, 2]`.

We have also provided a function called `cifar_grid`. This function will yield a 2 dimensional coordinate grid.
The domain for both dimensions is `[0, 32>`. By passing an interger to this function one can specify the resolution of this grid.
This function is particularly useful for sampling Cifar images hat varying resolutions.
```
some_grid = cifar_grid(64)
```
The example above would yield a grid where all coordinates lie between `[0.,0.]` and `[31.5, 31.5]`.

### Providing coordinate grids
Providing a coordinate grid to the models can be done in three different ways.
Firstly we could just provide the training coordinates when creating the model:
```
sine_MLP_INR = MLP_INRClass(train_coordinates=train_coordinates, **cfg_sine)
```
If the training coordinates are provided we can omit the `dim_linear` parameter, as it can be infered from the provided coordinate grid.

We could also set the training coordinates later on using the `set_train_coordinates()` method:
```
sine_MLP_INR.set_train_coordinates(train_coordinates)
```

Finally, we can also simply provide the training coordinates to the fit method:
```
sine_MLP_INR.fit(input_img, optimizer, criterion, scheduler, epochs, train_coordinates=train_coordinates)
```
Note that this method does not actually set the `.train_coordinates` attribute.

### Other parameters
If the training coordinate grid is provided (via one of the aforementioned methods) we can train start training the models.
To train our models one can call the `.fit` method. This method expects a target, optimizer, scheduler and a set of epoch sizes (as mentiond before, a coordinate grid may be provided).
```
from torchvision.datasets import CIFAR10
training_set = CIFAR10(transform=ToTensor(), root="data", download=True)
training_img = training_set.__getitem__(IMAGE_IDX)
input_img = training_img[0].permute((1,2,0))

cfg_sine = OmegaConf.load('./configs/RELU_SINE_INR_cfg.yaml')

domain = [[0,31][0,31]]
size = [32, 32]
train_coordinates = coordinate_grid(domain, size, reshape=False)

sine_MLP_INR = MLP_INRClass(train_coordinates=train_coordinates, **cfg_sine)
optimizer = torch.optim.Adam(sine_MLP1_INR.model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss(reduction='sum')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8)
epochs = [5]*200

sine_losses = sine_MLP1_INR.fit(input_img, optimizer, criterion, scheduler, epochs)
```
In this example a Siren is trained on the first image of the Cifar10 dataset. Note that this model trains for 200 epochs of 5 training steps.

## Testing
Once the model is fitted properly it will provide a continuous representation of its target image.
In order to sample from this representation a coordinate grid should be provided to the model:
```
domain = [[0,31][0,31]]
size = [64, 64]
test_coordinates = coordinate_grid(domain, size, reshape=False)

sample = sine_MLP1_INR(test_coordinates)
```
If no test coordinates are provided it will use the training coordinates (if they were also not provided an error message will pop up).

## Important Files
`INRBaseClass.py` defines the base class upon which the MLP (ReLU, Siren) and MFN (Gabor) INR models are build.  
`MLP_INRClass.py` defines the MLP INR class.  
`MLPClass` defines the MLP network that is used in the MLP INR.  
`Gabor_INRClass.py` defines the Gabor INR.  
`GaborClass.py` defines the Gabor network that is used in the Gabor INR.  
`helper_functions.py` contains the two functions used to create coordinate grids.  
`INR Tests.ipynb` provides a demo for using this repository.