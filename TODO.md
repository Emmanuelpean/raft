# To-Do List

## ✅ Completed

### Core Features
* Add the following data processing options:
  * [X] Fitting
  * [X] Derivative  
  * [X] Background subtraction  
  * [X] Normalisation (with respect to the smoothed data if present)
  * [X] Interpolation
  * [X] Averaging
* Added a toggle to reset the data processing settings when new files are loaded
* [X] Add interpolation for max/min point calculation  
* [X] Add an option to display or hide the raw data when smoothing is used.
* [X] Move "Download Processed Data" button at the end of the processing settings and add the smoothed and fitted data.
* [X] Add data processing and data extraction of 3D datasets:
  * [X] Add an option to select the z-axis data.
  * [X] Plot the extracted data vs the z-axis
  * [X] Add a file uploader allowing multiple files
* [X] Add a second tab to graphs to show the raw data

### UI & Branding
* [X] Added plot to explain how interpolation is used to improve the max/min point and FWHM calculations
* [X] Reduced the size of the file uploader
* [X] Increase plot text size
* [X] Add new logo to sidebar  
* [X] Update logo  

### Testing & Typing
* [X] Add type hints  

### Performance
* [X] Add caching for data loading  
* [X] Improve plot performance

### Bug Fixes  
* [X] Fix loading bug for diffrac and wire files  
* [X] Fix name error in License
* [X] Do not show FWHM if calculation fails  
* [X] Fix bug when switching fitting models  
* [X] Fix parameter value not updating when model changes
* [X] Delete temporary Wire files 

---

## 🔧 In Progress

### Display  
* [X] Add precision setting to `Dimension` class for display  
  * [X] When loading data, allow input of decimal places (e.g. Repeat 1 → 0 decimals)  
  * [X] When displaying results, auto-determine precision from calculation
* [X] Add colour map to 3D data  
* [X] Add hover templates with "filename" and "curve name"  
* [X] Add a list of example data to download.
* [X] Add proper ReadMe
* [X] Add tests for area calculation

### Processing
* [X] Automatically convert z_dict values
* [X] Deal with uneven data

### Testing  
* [X] Review and verify all tests  
* [X] Ensure ~100% test coverage 
* [X] Add test for z_dict value converting
* [X] Check that a failed test leads to a failed build
* [X] Check that .gitignore is working now

### Bug Fixes
* [X] Handle `NaN` values in `number_to_string`  
* [X] Fix the Reset Data Processing button
* [X] Split the utils module and add the session state submodule
* [X] Fix bug leading to missing curves when changing the data sorting key
* [X] Add pyproject file
* [X] Check if get_max is sensitive to nan and inf. functions return nan if failed to compute
* [X] Fix missing colouring in production
* [X] Fill data processing text
* [X] remove mentions of poetry
* [X] Update the changelog
* [X] Fix missing icon in production
* [X] Improve the fitting of multiple curves (display and store the p0 of each curve?)
---

## 🧩 Planned Features

### Interactive UI  
* [ ] Add draggable processing expanders
* [ ] Add tab to each plot for data display and download  
