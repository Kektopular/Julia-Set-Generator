# Julia-Set-Generator
Just a basic program using Python and TensorFlow, for generating and rendering Julia sets.

<br>

Some quick notes:

  I suspect that the CPU-only version of TensorFlow will be painfully slow, so suggest using the GPU version.
  
  The LOGDIR used in the code will have to be changed so that data can be written to disk on your setup.

  I found that the histogram summaries in TensorBoard are useful for designing color gradient functions, as can be seen further down.

<br><br>

A couple of Julia sets that were generated
![0.274 0.0063i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.274%200.0063i.jpg?raw=true)
![0.4 0.071i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.4%200.071i.jpg?raw=true)

<br><br>

The color gradient functions used for coloring these sets
![Color Gradient](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Color%20Gradient%20Functions.png?raw=true)

<br><br>

Visualization of the computational graph with TensorBoard
![Computational Graph](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Computational%20Graph.png?raw=true)

<br><br>

Histogram summaries of the divergence count and RGB values
![Histograms](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/TensorBoard%20Histograms.png?raw=true)

<br><br>

The Julia sets drawn as TensorBoard image summaries
![Images](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Tensorboard%20Images.png?raw=true)
