# MOSAICX
Multi-image Overlap Stitching and Automatic Construction of Coherent X-ray Imaging


these codes come in 4 parts in 4 different files that should be run in sequence

the preprocessing code is contained in: Dynmask

the first stitching code is contained in: MaxOverI

the second stitching code is contained in: MaxOverC

the postprocessing code is contained in: OpCr

the data that was used to build and test these codes are in the data file and can be used as an example. please extract and put into whichever folder you wish to test
you will find that the code has disreguarded columns 2 and 6, this is because those columns contained images to minimal features
to apply to another data set include columns. 

the code is split into two file types, the .py files which can be run as is in sequence 
and .ipynb files which have a few variations on the original methods that were used to test different functionality

it might be nessesary to change output and input directories/file names to get the code to run

there are a few dependancies, most notably CV2 which is used to handle image loading, saving, and transforms

for the .py files run them in sequence with the data folder moved into the .py folder

for the .ipynb files, move the data folder into the .ipynb folder. in the first file run all definitions and choose a cell that has the mode you are interested in running. run that cell. then move onto the next file in the sequence.

the file contained in the Alpha color folder was written by AI to quickly bridge the gap between the black and white and color images.

Starr Boney, Umeshika Dissanayaka, Lillian Rutowski, Aaron George, and Min Gyu Kim