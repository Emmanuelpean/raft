#### V 2.0.1 - May 2025
* 🌗 Added dark mode.

### V 2.0.0 - May 2025

##### ✨ New Data Processing Options
The following data processing options are now available:
* Averaging: if multiple signals are displayed, average every X signals.
* Fitting: apply a predefined function to fit the data.
* Derivative: compute the n<sup>th</sup>-order derivative of the data.
* Background subtraction: remove background by averaging the signal over a specified x-axis range.
* Normalisation: apply either maximum normalisation or feature scaling.
* Interpolation: interpolate the data using either the x-axis step size or the desired number of points.

##### 🌟 New Data Extraction Options
The following data extraction options are now available:
* Min/Max point interpolation: use cubic interpolation to improve the min/max point calculation.
* FWHM interpolation: use linear interpolation to improve the FWHM calculation.
* Area calculation: calculate the area under the curve.
* Precision control: users can now adjust the precision of extracted data.
* Data export: extracted data can now be downloaded for external use.

##### 🗃 New Data Upload Options
* Multi-file upload: multiple files can now be uploaded simultaneously.
* Batch processing: multiple signals can be processed and analysed at once.
* Signal sorting: when multiple signals are displayed, they can be sorted based on metadata extracted from the raw data 
  (*e.g.*, signal name, measurement timestamp, emission wavelength). The selected sorting option is also used to 
  generate the Z-axis data for plotting extracted and fitted parameters.
* Reset toggle: a toggle has been added to automatically reset data processing settings when new files are loaded.

##### 💻 Improved User Interface
* Data tab: each graph now includes a "Data" tab that displays the corresponding data in a table format.
* Improved readability: graph text size has been increased for better readability.
* Compact layout: the sidebar logo and file uploader have been resized to create a more compact layout.
* Data processing section: A new section has been added to provide a more detailed explanation of the data processing steps.
* Updated theme: the app theme has been refreshed to better match the Raft logo.
* Streamlined display: the main logo is now hidden when data is being displayed.

##### 👨‍💻 Code Quality & Performance Improvement
* Code refactoring: the script has been refactored for improved readability, structure, and maintainability.
* Testing coverage: added comprehensive tests, reaching 97% code coverage.

##### 🐛 Performance, Bug Fixes & Miscellaneous
* Data loading optimised: implemented caching to speed up data loading.
* Faster plotting: improved plotting performance for smoother interactivity.
* File reading fixes: resolved issues preventing proper reading of Diffrac and WiRE files.
* Licence correction: fixed a typo in the licence name.
* Error feedback: A message is now displayed when data extraction fails.
* Versioning reset: Version numbering has been reset for simplicity and clarity.

#### V 1.3.0 - February 2023 
* Bug fixes
* Now able to read incomplete Fluoracle files
  
#### V 1.2.0 - November 2022
* Data labelling has been significantly improved.
* Bug fix and code optimisation.
* Added Zem3 file support.

#### V 1.1.0 - October 2022
* Added the following options when a single data set is displayed:
    * Download button do download the data displayed.
    * Option to change the data range displayed.
    * Option to smooth the data using the Savitzky-Golay filter.
    * Option to determine the maximum and minimum point, and FWHM.

### V 1.0.0 - October 2022
* **Initial release**
