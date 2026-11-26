# PIG_Reflectance_Classifier
Application to help in classifying coloring materials according to their spectral-reflectance signatures.
## Installation
To use the app, download the _release_ folder and launch _PIG_Reflectance_Classifier.exe_. It takes time (1 min) but eventually the app appears. From there, you can create your very own classification of the coloring-material database, based on spectral criteria.  
To modify the app or launch it with Python, you can clone this repository and find the Python code in the _src_ folder.
## Usage 
Classify spectra according to their spectral signatures, compare new samples to the classification, add new samples in the database.


- Create classes by changing the threshold values of the criteria in the lower panel called _Spectral criteria_.  
The thresholds can be saved as .txt files using the _Save criteria_ button. If it is for classification purposes, the names of these files should start by _Class_.
These set of thresholds can be loaded again using the _Load criteria_ button.
- Save the classification results (spectra per class, figures showing each class, envi files of each class and more...) using the _Save classes_ button.
- Load a new sample, compare it to the classification you made and/or save it into the database using the _Add sample_ panel.
- The app can be used on different spectral databases by modifying the R_nowater.xlsx and R.xlsx files in the data folder.
## Contributors
Fanny Lefaucher, Bernard Schmitt, Emilie Chalmin, Helene Salomon, and many other collaborators who helped in performing the measurements.

