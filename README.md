# Julia-Set-Generator
Just a basic program written in Python and using TensorFlow, for generating and rendering Julia sets. It collects a complex number for c in z_n+1 = z^2 + c with user input, which defines the set that is generated. You can easily change the domain, range, resolution, and color gradient through simple modifications of the code as well.

<br>

Some quick notes:

  I suspect that the CPU-only version of TensorFlow will be painfully slow, so suggest using the GPU version.
  
  The LOGDIR used in the code will have to be changed so that data can be written to disk on your setup.

  I found that the histogram summaries in TensorBoard are useful for designing color gradient functions, as can be seen further down.

<br><br>

Some of my personal favorites that were made visible thanks to this program
![-0.764 0.12i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%20-0.764%200.12i.jpg?raw=true)
![-0.835 0.22i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%20-0.835%200.22i.jpg?raw=true)
![0.0 0.74i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.0%200.74i.jpg?raw=true)
![0.285 0.012i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.285%200.012i.jpg?raw=true)

<br><br>

A couple more Julia sets that were generated, and the color gradient used for coloring them
![0.274 0.0063i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.274%200.0063i.jpg?raw=true)
![0.4 0.071i](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Julia%20Set%200.4%200.071i.jpg?raw=true)
![Color Gradient](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Color%20Gradient%20Functions.png?raw=true)

<br><br>

Visualization of the computational graph with TensorBoard
![Computational Graph](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Computational%20Graph.png?raw=true)

<br><br>

Histogram summaries of the divergence count and RGB values for the second set
![Histograms](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/TensorBoard%20Histograms.png?raw=true)

<br><br>

The sets drawn as TensorBoard image summaries
![Images](https://github.com/Kektopular/Julia-Set-Generator/blob/master/Example%20Images/Tensorboard%20Images.png?raw=true)
