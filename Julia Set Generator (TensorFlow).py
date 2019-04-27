# -*- coding: utf-8 -*-
"""
Julia Set Generator (TensorFlow)
June 2018 (Updated: April 2019)
Real-Time Chris
https://github.com/RealTimeChris
"""



# Import libraries and stuff.
import datetime as dt
import numpy as np
import tensorflow as tf
import PIL as PIL



# Image Dimensions.
image_width = 7680
image_height = 4320

# Set the LOG_DIR for storing TensorBoard log data.
LOG_DIR = ("")

# Set the IMG_DIR for storing each of the generated Julia Set images.
IMG_DIR = ("")

# Create a datetime object for organizing TensorBoard data.
date_time = str(dt.datetime.strftime(dt.datetime.today(), "%Y-%m-%d %H-%M-%S"))

# Reset the default graph.
tf.reset_default_graph()

# Create a FileWriter object for writing TensorBoard data to disk.
writer = tf.summary.FileWriter(
        LOG_DIR + date_time,
        graph = tf.get_default_graph())

# Create objects for collecting session data.
run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Create an object for running TensorFlow ops.
session = tf.Session()



# Create the section of the complex plane to be used.
with tf.variable_scope("complex_plane_setup"):
    
    # Set up the domain and range.
    domain_set =  tf.range(
            -2.13,
            limit = 2.13,
            delta = (4.26 / image_width),
            dtype = tf.float64,
            name = "domain_set")
    domain_reshape = tf.reshape(domain_set, (1, image_width), name = "domain_reshape")
    domain_tile = tf.tile(domain_reshape, [image_height, 1], name = "domain_tile")
    
    range_set = tf.range(
            -1.2,
            limit = 1.2,
            delta = (2.4 / image_height),
            dtype = tf.float64,
            name = "range_set")
    range_reshape = tf.reshape(range_set, (image_height, 1), name = "range_reshape")
    range_reverse = tf.reverse(range_reshape, [0], name = "range_reverse")
    range_tile = tf.tile(range_reverse, [1, image_width], name = "range_tile")
    
    # Combine the domain and range into a section of the complex plane.
    complex_plane = tf.complex(domain_tile, range_tile, name = "complex_plane")


# Collect the c-value for choosing which Julia set to generate.
with tf.variable_scope("c_setup"):
    
    # Collect the real part of the c-value.
    c_real_raw = float(input("Enter the real part of the c-value: \n"))
    c_real = tf.constant(c_real_raw, dtype = tf.float64, name = "c_real")
    
    # Collect the imaginary part of the c-value.
    c_i_raw = float(input("Enter the imaginary part of the c-value: \n"))
    c_imaginary = tf.constant(c_i_raw, dtype = tf.float64, name = "c_imaginary")
    
    # Combine the real and imaginary parts into a complex c-value.
    c = tf.complex(c_real, c_imaginary, name = "c")


# Create the variable matrices for holding the z-value and the divergence counts.
with tf.variable_scope("variable_setup"):
    
    # The matrix of z-values for each iteration of z_n+1 = z^2 + c.
    z = tf.Variable(initial_value = complex_plane, name = "z")
    
    # The matrix of iteration counts that it takes for each pixel-value to diverge.
    div_count = tf.get_variable(
            "div_count",
            shape = (image_height, image_width),
            dtype = tf.float64,
            initializer = tf.zeros_initializer(dtype = tf.float64))
    
    # Attach some summaries to the divergence count.
    div_count_histogram = tf.summary.histogram("div_count_hist", div_count)
    
    div_count_img_prep = tf.reshape(
            div_count,
            (1, image_height, image_width, -1),
            name = "div_count_img_prep")
    div_count_img = tf.summary.image("div_count_img", div_count_img_prep)
    
    experiment_summaries_op = tf.summary.merge_all()


# Initialize global variables before using them in operations.
session.run(
        tf.global_variables_initializer(),
        options = run_options,
        run_metadata = run_metadata)
writer.add_run_metadata(run_metadata, "init_1")


# Create an operation for the the function z_n+1 = z^2 + c.
with tf.variable_scope("z_n"):
    z_squared = tf.square(z, name = "z_squared")
    plus_c = tf.add(z_squared, c, name = "plus_c")
    z_n_assign = z.assign(plus_c)

# Create the operation for tracking divergence counts.
with tf.variable_scope("div_check"):
    bound_value = tf.constant(2, dtype = tf.float64, name = "bound_value")
    div_check = tf.cast(
            tf.abs(z, name = "z_abs") < bound_value,
            dtype = tf.float64,
            name = "div_check")
    div_count_assign = div_count.assign_add(div_check)

# Group the operations together for efficient execution.
with tf.variable_scope("experiment_ops"):    
    experiment_ops = tf.group(
            z_n_assign,
            div_count_assign)


# Generate the set.
for i in range(1,101):
    
    # Denote that something is happening.
    print("Calculating " +  str(image_width * image_height) +
          " instances of z_n+1 = z^2 + c (Iteration: " + str(i) + ")\n")
    
    # Run the set operations.
    session.run(
            experiment_ops,
            options = run_options,
            run_metadata = run_metadata)
    
    
    # Lines 165-178, inclusive, can be commented out to speed up execution if desired.
    
    # Add the experiment op runtime stats to the TensorBoard event file.
    writer.add_run_metadata(run_metadata, "exp_ops_" + str(i))
    
    # Run the experiment summary ops.
    experiment_summaries = session.run(
            experiment_summaries_op,
            options = run_options,
            run_metadata = run_metadata
            )
    
    # Add the experiment summary op runtime stats to the TB event file.
    writer.add_run_metadata(run_metadata, "exp_summaries_" + str(i))
    
    # Add the experiment summaries to the TB event file.
    writer.add_summary(experiment_summaries)
    


# It's done.
print("Done!")

# Flush the TensorBoard data to disk.
writer.flush()


# Use the resultant divergence count matrix to render an image of the set.
with tf.variable_scope("image_rendering"):

    # Reshape the divergence count matrix so that it exists in 3 dimensions.
    div_count_reshape = tf.reshape(
            div_count,
            (image_height, image_width, 1),
            name = "div_count_reshape")

    # Create the red RGB layer.
    with tf.variable_scope("red_transform"):
        red_d = tf.constant(0, dtype = tf.float64, name = "red_d")
        red_k = tf.constant(0.314, dtype = tf.float64, name = "red_k")
        red_a = tf.constant(20, dtype = tf.float64, name = "red_a")
        red_c = tf.constant(10, dtype = tf.float64, name = "red_c")
        
        red_h_shift = tf.subtract(div_count_reshape, red_d, name = "red_h_shift")
        red_h_scale = tf.multiply(red_h_shift, red_k, name = "red_h_scale")
        red_fn = tf.sin(red_h_scale, name = "red_fn")
        red_v_scale = tf.multiply(red_fn, red_a, name = "red_v_scale")
        red_v_shift = tf.add(red_v_scale, red_c, name = "red_v_shift")
        
        # Attach a histogram summary to the red layer.
        red_histogram = tf.summary.histogram("red_layer", red_v_shift)
        
    # Create the green RGB layer.
    with tf.variable_scope("green_transform"):
        green_d = tf.constant(4.7, dtype = tf.float64, name = "green_d")
        green_k = tf.constant(0.314, dtype = tf.float64, name = "green_k")
        green_a = tf.constant(50, dtype = tf.float64, name = "green_a")
        green_c = tf.constant(30, dtype = tf.float64, name = "green_c")
        
        green_h_shift = tf.subtract(div_count_reshape, green_d, name = "green_h_shift")
        green_h_scale = tf.multiply(green_h_shift, green_k, name = "green_h_scale")
        green_fn = tf.sin(green_h_scale, name = "green_fn")
        green_v_scale = tf.multiply(green_fn, green_a, name = "green_v_scale")
        green_v_shift = tf.add(green_v_scale, green_c, name = "green_v_shift")
        
        # Attach a histogram summary to the green layer.
        green_histogram = tf.summary.histogram("green_layer", green_v_shift)
        
    # Create the blue RGB layer.
    with tf.variable_scope("blue_transform"):
        blue_d = tf.constant(0, dtype = tf.float64, name = "blue_d")
        blue_k = tf.constant(0.314, dtype = tf.float64, name = "blue_k")
        blue_a = tf.constant(-80, dtype = tf.float64, name = "blue_a")
        blue_c = tf.constant(155, dtype = tf.float64, name = "blue_c")
        
        blue_h_shift = tf.subtract(div_count_reshape, blue_d, name = "blue_h_shift")
        blue_h_scale = tf.multiply(blue_h_shift, blue_k, name = "blue_h_scale")
        blue_fn = tf.sin(blue_h_scale, name = "blue_fn")
        blue_v_scale = tf.multiply(blue_fn, blue_a, name = "blue_v_scale")
        blue_v_shift = tf.add(blue_v_scale, blue_c, name = "blue_v_shift")
        
        # Attach a histogram summary to the green layer.
        blue_histogram = tf.summary.histogram("blue_layer", blue_v_shift)
        
    # Merge the color layer ops.
    with tf.variable_scope("color_layer_ops"):
        color_layers = tf.group(
                [red_v_shift, green_v_shift, blue_v_shift],
                name = "color_layers")

    # Merge the image rendering summaries.
    color_layer_summaries_op = tf.summary.merge(
            [red_histogram,
            green_histogram,
            blue_histogram], name = "color_layer_summaries_op")


# Add the graph to the TensorBoard event file.
writer.add_graph(tf.get_default_graph())


# Execute the color layer ops.
session.run(
        color_layers,
        options = run_options,
        run_metadata = run_metadata)

# Write the color layer runtime stats to the TensorBoard event file.
writer.add_run_metadata(run_metadata, "color_layers")

# Execute the color layer summaries op.
color_layer_summaries = session.run(
        color_layer_summaries_op,
        options = run_options,
        run_metadata = run_metadata)

# Write the color layer summary runtime stats to the TensorBoard event file.
writer.add_run_metadata(run_metadata, "color_layer_summaries")

# Write the color layer summaries to the TensorBoard event file.
writer.add_summary(color_layer_summaries)

# Flush the TensorBoard data to disk.
writer.flush()


# Convert the divergence count matrix into a numpy array (for setting max = 0).
div_count_np = np.array(session.run(div_count_reshape))

# Set the contained values to 0 (Black).
div_count_red = np.array(session.run(red_v_shift))
div_count_red[div_count_np == div_count_np.max()] = 0

div_count_green = np.array(session.run(green_v_shift))
div_count_green[div_count_np == div_count_np.max()] = 0

div_count_blue = np.array(session.run(blue_v_shift))
div_count_blue[div_count_np == div_count_np.max()] = 0

# Concatenate the RGB layers into a single RGB array.
div_count_rgb = np.concatenate(
        [div_count_red, div_count_green, div_count_blue],
        2)

# Convert the pixel color values into unsigned 8-bit integers and clip them.
div_count_uint = np.uint8(np.clip(div_count_rgb, 0, 255))

# Create an Image from the array of pixel color values.
img = PIL.Image.fromarray(div_count_uint, mode = "RGB")

# Save the Image to a folder as a jpg.
img.save(
        IMG_DIR + str(c_real_raw) + " " +
        "+" + " " + str(c_i_raw) + "i.jpg")

# Close the TensorFlow session.
session.close()




