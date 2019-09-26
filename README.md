# Julia Set Generator (TensorFlow)

This is a basic program written in Python and using TensorFlow, for generating and saving Julia sets. While I was first learning Python, I came across TensorFlow and TensorBoard and thought that they were interesting looking tools. I am also quite fond of visualizations, so I thought that this would be a fun thing to make, and it was!

I originally created this program in May 2018, but in April 2019 I decided to get it working again and upload it fresh to GitHub.

It collects a complex number for c in z_(n+1)=z_n^2+c with user input, which defines the set that is generated. You can easily change the domain, range, resolution, and color gradient through simple modifications of the code as well.

First, here is the Development Environment and Installation Guide:

https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Development%20Environment%20and%20Installation%20Guide.md

Then, here is a short Quick Start Guide:

https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Quick%20Start%20Guide.md

Some quick notes:
  
  -I suspect that the CPU-only version of TensorFlow will be painfully slow, so suggest using the GPU version.
  
  -The LOG_DIR and IMG_DIR used in the code will have to be changed so that data can be written to disk on your setup.

<br>

Some of my personal favorites that were made visible using this program:
<br>
-0.835 + 0.22i
![-0.835 + 0.22i](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/-0.835%20+%200.22i.jpg?raw=true)
0.285 + 0.012i
![0.285 + 0.012i](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/0.285%20+%200.012i.jpg?raw=true)
-0.764 + 0.1185i
![-0.764 + 0.1185i](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/-0.764%20+%200.1185i.jpg?raw=true)

<br>

A couple more sets that were generated, and the color channel mapping functions used for coloring them:
<br>
0.274 + 0.0063i
![0.274 + 0.0063i](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/0.274%20+%200.0063i.jpg?raw=true)
0.4 + 0.071i
![0.4 0.071i](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/0.4%20+%200.071i.jpg?raw=true)
![Color Channel Mapping Functions:](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/Color%20Channel%20Mapping%20Functions%20-%20The%20Blue.png?raw=true)

<br>

Visualization of the computational graph with TensorBoard:
![Computational Graph Visualization](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/Computational%20Graph%20Visualization.png?raw=true)

<br>

TensorBoard Histogram Summaries:
![Histogram Summaries](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/Histogram%20Summaries.png?raw=true)

<br>

TensorBoard Image Summaries:
![Image Summaries](https://github.com/RealTimeChris/Julia-Set-Generator-TensorFlow/blob/master/Images/Image%20Summaries.png?raw=true)
