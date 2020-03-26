# Echelon
A multi tiered neural network for static malware detection

# http://localhost:8888/notebooks/PE_Section_Data_Extractor.ipynb#

# Issues In-progress:
Feature Map:
* Sections are not covering all the parts of an executable file
* Count of activations or Sum of magnitudes of activations
* Section is found - but data is empty (Section_size_byte = 0) and end_offset = -1
* How to concatenate sections - Insert dummy window of size 500 bytes between all sections to be concatenated?
* Check OFFSET leakage
* Check validation in TIER-2: currently not performed

Challenges:
* The problem of Coverage:
- Does the highly qualified sections cover all samples in training and testing data?
i.e., the number of samples intersected by the selected sections should be equal to 'U'

# QA:

* If section size is smaller than convolution window size [500 bytes] - Reset window size to a size less than the minimum section size.
[Caveat: Blows up memory requirement]