# empirical_early_prediction
This repository contains the Python scripts and processed data required to recreate the results presented in the paper: “Coupling a Capacity Fade Model with Machine Learning for Early Prediction of the Battery Capacity Trajectory”, published in Applied Energy (DOI: )  
*Questions about the scripts and data provides in this repository can be directed to Tingkai Li ([tingkai.li@uconn.edu](mailto:tingkai.li@uconn.edu))*

## Paper Details
### Coupling a Capacity Fade Model with Machine Learning for Early Prediction of the Battery Capacity Trajectory
Tingkai Li $^{1}$, Jinqiang Liu $^2$, Adam Thelen $^3$, Ankush Kumar Mishra $^3$, Xiao-Guang Yang $^{4,*}$, Zhaoyu Wang $^2$, Chao Hu $^{1,\dagger}$

$^{1}$ School of Mechanical, Aerospace, and Manufacturing Engineering, University of Connecticut, Storrs, CT, 06269, USA  
$^{2}$ Department of Electrical and Computer Engineering, Iowa State University, Ames, IA, 50011, USA  
$^{3}$ Department of Mechanical Engineering, Iowa State University, Ames, IA, 50011, USA  
$^4$ Department of Mechanical Engineering and Electrochemical Engine Center, Pennsylvania State University, University Park, PA 16802, USA  
$^*$ Present Affiliation: School of Mechanical Engineering, Beijing Institute of Technology, Beijing, 100081, China  
$^{\dagger}$ Indicates corresponding author.

#### Abstract
<p style="text-align: justify;">Early prediction of battery capacity degradation, both the end of life and the entire degradation trajectory, can accelerate aging-focused manufacturing and design problems. 
However, most state-of-the-art research on early capacity trajectory prediction focuses on developing purely data-driven approaches to predict the capacity fade trajectory of cells, sometimes leading to overconfident models that generalize poorly. 
This work investigates three methods of integrating empirical capacity fade models into a machine learning framework to improve the model's accuracy and uncertainty calibration when generalizing beyond the training dataset. 
A critical element of our framework is the end-to-end optimization problem formulated to simultaneously fit an empirical capacity fade model to estimate the capacity trajectory and train a machine learning model to estimate the parameters of the empirical model using features from early-life data. 
The proposed end-to-end learning approach achieves prediction accuracies of less than 2% mean absolute error for in-distribution test samples and less than 4% mean absolute error for out-of-distribution samples using standard machine learning algorithms.
Additionally, the end-to-end framework is extended to enable probabilistic predictions, showing the model uncertainty estimates are appropriately calibrated, even for out-of-distribution samples.

![Figure 1: An overview of the early trajectory prediction problem studied in this work.](https://github.com/tingkai-li/empirical_early_prediction/Figure1.png)
*Figure 1: Overview of the early trajectory prediction problem studied in this work.*  
![Figure 2: Overview of capacity trajectory prediction approaches. (a) Knot point-based prediction of empirical model parameters; (b) sequential prediction of empirical model parameters; (c) end-to-end prediction of empirical model parameters.](https://github.com/tingkai-li/empirical_early_prediction/Figure2.png)
*Figure 2: Overview of capacity trajectory prediction approaches. (a) Knot point-based prediction of empirical model parameters; (b) sequential prediction of empirical model parameters; (c) end-to-end prediction of empirical model parameters.*
</p>

## Repository Structure
```
|- Case_study_LSTM_ensemble/
    |- Data/
    |- Results/
|- Case_study_NCA_dataset
    |- Processed_data/
|- Data_preprocessing
|- Empirical_model_fitting
|- Main_study
    |- Best_network/
    |- Empirical_parameter_results/
    |- Processed_input_output/
|- Parametric_study_feature_week
    |- Data_preprocessing/
    |- Predicted_parameters/
|- Figure1.png
|- Figure2.png
|- LICENSE
|- README.md
```
### `Case_study_LSTM_ensemble/`
This directory contains Jupyter notebooks and processed data for the benchmarking case study presented in *Section 6.1* of the paper. The method is originally proposed in "*Uncertainty-aware and explainable
machine learning for early prediction of battery degradation trajectory*" by Rieger et al. (DOI: [10.1039/D2DD00067A](https://doi.org/10.1039/D2DD00067A))  

* `Data/`: Contains data processed for LSTM ensemble after running `preprocess_data_SOTA.ipynb`
* `Results/`: Contains prediction results from LSTM ensemble after running `case_study.ipynb`
* `case_study.ipynb`: Provides the implementation of the LSTM ensemble on the ISU-ILCC dataset
* `CRPS_df_e2e.csv`: CRPS score from end-to-end ensemble, for visualization purposes
* `ECE_df_e2e.csv`: ECE score from end-to-end ensemble, for visualization purposes
* `MAE_df_e2e.csv`: MAE score from end-to-end ensemble, for visualization purposes
* `model_state_dict.pth`: Saved model parameters from the LSTM ensemble method after running `case_study.ipynb`
* `preprocess_data_SOTA.ipynb`: Provides data preprocessing for the LSTM ensemble method
* `visualization_evaluation.ipynb`: Visualizes the prediction results and compares with the end-to-end ensemble method

### `Case_study_NCA_dataset/`
This directory contains Jupyter notebooks and processed data for the benchmarking case study presented in *Section 6.2* of the paper. The original dataset can be accessed through Figshare (DOI: [10.6084/m9.figshare.25975315](https://doi.org/10.6084/m9.figshare.25975315)) and the data discriptor can be found on *Scientific Data 11, 1020 (2024)* (DOI: [10.1038/s41597-024-03859-z](https://doi.org/10.1038/s41597-024-03859-z)).

* `Processed_data`: This directory contains interpolated curves from the data
* `ah_throughput_test.csv`: Ah-throughput of equidistant SOC levels for cells in the test subset after running `data_preprocessing.ipynb`
* `ah_throughput_train.csv`: Ah-throughput of equidistant SOC levels for cells in the training subset after running `data_preprocessing.ipynb`
* `case_study.ipynb`: Implementation of end-to-end method based on elastic net regression on the NCA dataset
* `feature_test_scaled.csv`: Early-life features for the test subsets after transforming using a StandardScaler, obtained after running `data_preprocessing.ipynb`
* `feature_test.csv`: Early-life features for the test subset after running `data_preprocessing.ipynb`
* `feature_train_scaled.csv`: Early-life features for the training subsets after transforming using a StandardScaler, obtained after running `data_preprocessing.ipynb`
* `feature_train.csv`: Early-life features for the training subset after running `data_preprocessing.ipynb`
* `trajectory_Ah_throughput_interp_test.csv`: Concatenated table for the  capacity trajectories of all test cells after interpolation, obtained after running `data_preprocessing.ipynb`
* `trajectory_Ah_throughput_interp_train.csv`: Concatenated table for the  capacity trajectories of all training cells after interpolation, obtained after running `data_preprocessing.ipynb`
* `visualization.ipynb`: Visualizes the capacity trajectories for both subsets
* `worker_NCA.py`: Worker for parallel computing used by `case_study.ipynb`

### `Data_processing/`
This directory contains preprocessed data in CSV format and a Juputer notebook `trajectory_interpolation.ipynb` generates all these data files.

### `Empirical_model_fitting`
This directory contains fitted empirical model parameters using a bi-level curve fitting strategy. The data files in CSV format is obtained after running the Jupyter notebook `empirical_fitting_python.ipynb`

### `Main_study`
This directory contains Juputer notebooks for the three different prediction methods on the ISU-ILCC dataset, which is available through *Iowa State University DataShare* (DOI: [10.25380/iastate.22582234](https://doi.org/10.25380/iastate.22582234)).
* `Best_network/`: This directory contains the model parameters from all neural network based methods presented in the manuscript for reproducability
* `Empirical_parameter_results/`: This directory contains the predicted empirical model parameters from all methods for visualization purposes
* `Processed_input_output/`: This directory contains most of processed data needed to train different models. These data files are obtained after running `input_output_preparation.ipynb`, which has all necessary data transformation steps.
* `calibration_vs_num_models.ipynb`: Provides all necessary post-analysis on the end-to-end ensemble method, including ensembling predictions, calculating evaluation metrics, and visualizing the calibration curves. 
* `end_to_end_ensemble.ipynb`: The implementation of the end-to-end approach with neural network ensembles
* `end_to_end.ipynb`: The implementation of end-to-end approach with elastic net regression and MLP neural networks
* `generate_CV.ipynb`: Provides a universal 10-fold cross-validation dataset for all implementations
* `input_output_preparation.ipynb`: Generated processed input and output data files needed for all implementations.
* `knot_points.ipynb`: Implementation of the knot point-based approach with MLP neural networks.
* `sequential_optimization.ipynb`: The implementation of the sequential optimization approach with elastic net regression and MLP neural networks
* `training_cells_CV.pkl`: Contains the training partitions of 10-fold cross-validation
* `val_cells_CV.pkl`: Contains the validation partitions of 10-fold cross-validation

### `Parametric_study_feature_week/`
This directory contains a Jupyter notebook and data files for the parametric study presented in *Section 5.6* of the paper.  
* `Data_processing/`: This directory contains a Python script `early_life_feature_for_week_study.py` for feature extraction, a Jupyter notebook `prepare_input_output.ipynb` for feature processing and label generation, and all data files in CSV format.
* `Predicted_parameters/`: This directory contains the predicted empirical model parameters after running `weekly_study_e2e_enr.ipynb` for visualization.
* `b_scatter_plot_weekly.m`: Provides visualization of the distribution of predicted empirical model parameters
* `best_result_weekly_study.pkl`: Contains the best weights for the elastic net regression models for each week
* `results_weekly_study.pkl`: Contains the optimization results of multistart optimization for training the end-to-end models
* `weekly_study_e2e_enr.ipynb`: The implementation of the end-to-end approach with elastic net regression for the parametric study