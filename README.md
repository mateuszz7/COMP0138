# Investigating Uncertainty Estimation in Forecasting Influenza

This repository holds all the code used/modified/created for my dissertation titled "Investigating Uncertainty Estimation in Forecasting Influenza", as part of the MEng Computer Science degree at University College London (UCL).

The repository is mainly structured where a folder corresponds to a chapter of the report (and the notebooks inside correspond to sections of the report), with exception of 4 folders:
- `common` - folder contains common functions and variables used for code in defferent chapters
- `michael_morris_github_code` - this is a copy of [Morris et al. repository](https://github.com/M-Morris-95/Forecasting-Influenza-Using-Neural-Networks-with-Uncertainty) at the time of writing this, I make calls to some of the functions contained in this folder but also re-use some code and explicitly refer to the original file in a file that does this so that a diff can be ran between the files to see the modifications.
- `michael_morris_email_code` - this is a copy of code that was shared to me by Morris et al. in January 2023, containing experimental code they tried but did not share elsewhere, similar to the previous folder, I make calls to some of the functions contained in this folder and explicitly refer to the original file in a file that does this so that a diff can be ran between the files to see the modifications.
- `chapters_5_6_7_8` - this contains the code used to make and run all the iterative neural networks used in the report as well as hyperparameter optimisation and generation of data for this.

In addition to the code from Morris et al. I also use libraries for various parts of the report, below I summarise which I use and what for:

|Library|Part of library used|Used for|Link to documentation|
| :---------------- | :------ | :---- | :---- |
| NumPy | Multiple parts | Various matrix and vector operations required for machine learning | [https://numpy.org/doc/](https://numpy.org/doc/) |
| TensorFlow | Multiple parts | Mainly to build, train and run neural networks  | [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs) |
| TensorFlow Probability | Multiple parts | Probabilistic parts of the neural networks I built with TensorFlow (see above), e.g. distributional layers | [https://www.tensorflow.org/probability/api_docs/python/tfp](https://www.tensorflow.org/probability/api_docs/python/tfp) |
| SciPy | stats | Statistical calculations, e.g. probability density | [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/) |
| matplotlib | pyplot | All graphing/visualisation | [https://matplotlib.org/stable/index.html](https://matplotlib.org/stable/index.html) |
| multiprocessing | Multiple parts | Parallelism to run models in different processes (to utilise all processors) | [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html) |
| platform | processor | Detect the CPU architecture and set TensorFlow parameters to run accordingly | [https://docs.python.org/3/library/platform.html](https://docs.python.org/3/library/platform.html) |
| sys | path | Handling paths to access files | [https://docs.python.org/3/library/sys.html](https://docs.python.org/3/library/sys.html) |
| scikit-learn | Multiple parts | Calculating various metrics and for regression | [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) |
| properscoring | ps.crps_gaussian | Calculating CRPS | [https://github.com/properscoring/properscoring](https://github.com/properscoring/properscoring) |
| Keras | Multiple parts | Building customised layers | [https://keras.io/api/](https://keras.io/api/) |
| datetime | Multiple parts | Handling date types | [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html) |
| pandas | Multiple parts | Manipulating json data easier | [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) |
| json | Multiple parts | Handling interaction with json files | [https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html) |
| Bayesian Optimization | bayes_opt.BayesianOptimization | Bayesian hyperparameter optimisation | [https://github.com/fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization) |
| Evidential Deep Learning | layers, losses | Layer and loss for evidential deep learning | [https://github.com/aamini/evidential-deep-learning](https://github.com/aamini/evidential-deep-learning) |
| TyXe | Multiple parts | To build and train a fully Bayesian neural network | [https://github.com/TyXe-BDL/TyXe](https://github.com/TyXe-BDL/TyXe) |
| Pyro | Multiple parts | Run a a fully Bayesian neural network using TyXe (see above)| [https://docs.pyro.ai/en/stable/](https://docs.pyro.ai/en/stable/) |
| PyTorch | Multiple parts | To build and train neural networks | [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html) |
| dill | load, dump | To save neural networks | [https://dill.readthedocs.io/en/latest/](https://dill.readthedocs.io/en/latest/) |

### References
[1] Michael Morris, Peter Hayes, Ingemar J. Cox, and Vasileios Lampos. Forecasting influenza using neural networks with uncertainty. Paper is currently under review to be published in Nature Scientific Reports, a draft copy, obtained from Morris et al. on 21st September 2022, can be found in the appendix J of the report.