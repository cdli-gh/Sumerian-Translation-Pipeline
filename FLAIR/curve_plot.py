# Just replace the names of model and print the details of training and weights

from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('FLAIR/resources/taggers/flairpos1/loss.tsv')
plotter.plot_weights('FLAIR/resources/taggers/flairpos1/weights.txt')
