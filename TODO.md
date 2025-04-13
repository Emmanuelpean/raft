# To-Do List

## Feature Enhancements
* [X] Increase the size of the plot text.
* [X] Add fitting options.
* [X] Add interpolation to max/min point calculation
* [X] Added background removal
* [X] Add normalisation options
* [X] Reset settings for different data types
* [X] Display visual graphs explaining the use of interpolation
* [X] Disable the feature scaling options instead of hiding them
* [X] Improve the number_to_string function and split between display and input
* [X] Normalise the smoothed signal
* [X] Move "Download Data" to the bottom and include the smoothed data and fitted
* [X] Add interpolation
* [X] Add derivative
* [X] Update description
* [X] Add default guess values if guess value function fails
* [ ] Add precision to Dimension class for display purpose
  * [ ] When loading data files, input the number of decimals (e.g. Repeat 1, 0 decimals)
  * [ ] When displaying results, determine the precision from the calculation?
* [ ] Delete temporary files Wire

## Tests
* [X] Add tests
* [X] Add type hints
* [X] Add test for to_scientific new n parameter
* [ ] Check tests

## Bug Fixes
* [X] Fix bug preventing the loading of diffrac and wire files
* [X] Fix name error in License
* [X] Fix Max. wavenumber (cm<sup>-1</sup>) in range label
* [X] FWHM should not be displayed if could not be calculated
* [X] When changing the model, the value of the selected parameter does not change
* [ ] Review plot module and functions
* [X] Bug when changing fitting model
* [ ] Handle nan values in number to string.

## Future Features 
* [ ] Add draggable processing expanders.
* [ ] Prevent graph from changing by storing figure in session state.


## V 3.0
* [ ] Add support for 3D data
  * [ ] Allow user to select a z_dict quantity to the z-axis or by default use step of 1
  * [ ] Plot the data extracted vs the z-axis
  * [ ] Priority: Add caching for data loading
