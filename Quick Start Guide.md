To generate and save a Julia Set:

	  1. Select a directory on your computer to save the TensorBoard log data, and set this directory on Line 23.

	  2. Select a directory on your computer to save the generated Julia Set images, and set this directory on Line 28.

	  3. Run "Julia Set Generator (TensorFlow).py"

	  4. Enter the real and imaginary components of the complex number for selecting which Julia Set to generate.
		    For example: 0.274 and 0.0063

	  5. Wait for it to complete, and check the image directory!


To explore the program/Julia Set generation using TensorBoard:

	  1. Enter the following in command prompt/powershell: python -m tensorboard.main --logdir 'YOUR LOG_DIR HERE' --host localhost --port 8088

      2. Direct your internet browser to this URL: http://localhost:8088

	  3. Enjoy!
