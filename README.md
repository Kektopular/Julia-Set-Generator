# Julia-Set-Generator
Just a basic program using Python and TensorFlow, for generating and rendering Julia Sets.


Some quick notes to get things running

  The numpy and PIL libraries are required

  TensorFlow must be installed and running - I suspect that the CPU-only version will be painfully slow, so suggest using the GPU version. Likely there will be an error requiring that cupti64_90.dll be relocated - easy fix though, just copy it over.

  The LOGDIR used in the code will have to be changed so that data can be written to disk on your setup.

  I found that the histogram summaries in TensorBoard are useful for designing color gradient functions, as can be seen in the images.

