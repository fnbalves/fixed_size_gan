A fixed size GAN architecture. The size of the network does not
depend on the number of image classes. It uses ideas from Zero Shot Learning
networks (for image recognition).

Files:

* train_fixed_size_gan_info.py: Trains a fixed size GAN using the InfoGAN approach

* train_fixed_size_gan_minibatch.py: Trains a fixed size GAN using the Minibatch discriminator approach

* train_fixed_size_gan_new_minibatch.py: Trains a fixed size GAN using a modified Minibatch discriminator approach

* generate_gan_results_for_class.py: Creates sample images with a GAN network for a specific class

* train_multiple_zero_shot_nets.py: Trains two Zero Shot networks. One using the loss used at the DeViSE model and other using a new loss function

* create_performance_plots.py: Compares the performance of the two losses mentioned on the previous item. In order to use this file, you must
run the train_multiple_zero_shot_nets selecting 20, 40, 60 and 80 known classes so the required files will be created (this may take a long time)

* visualize_zero_shot_results.py: Creates TSNE visualizations and accuracy histograms for a zero shot learning network
