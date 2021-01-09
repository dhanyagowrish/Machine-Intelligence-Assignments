README

IMPLEMENTATION:

Pre-Processing Implementation:

Missing Values Imputation:

- 'Age' : Missing values imputed with column mean 
- 'Weight' :  column mean (not considering outliers in the mean calculation)
- 'Delivery phase' : missing values imputed with '1' . Cells with '2' replaced with '0'
- 'HB' : column mean
- 'BP' : column mean (not considering outliers in the mean calculation)
- 'Education' : missing values imputed with '5'
- 'Residence' - missing values imputed with '1'. Cells with '2' replaced with '0'

* 'Community' column was One-Hot encoded

Normalization of Columns:

-'Age', 'Weight', 'BP', 'HB' were normalized to have mean=0 and standard deviation=1

- 'Education' feature was normalized using Min-Max normalization technique


NEURAL NETWORK:

- Input Layer has 12 neurons ( 12 dimensional input feature vector )
- The neural network has 1 Hidden Layer
- Hidden Layer has 14 neurons and the activation function is ReLu
- Output Layer has 1 neuron and the activation function is Sigmoid

- Loss function used : Cross Entropy loss
- Learning Rate : 0.001
- Optimizer : Gradient Descent
- The model was trained for 800 iterations

- The weight and bias matrices were intialialized with random values sampled from normal Gaussian distribution. The random function was seeded to 1. 

- The data was split into Train and Test sets in the ratio 70 : 30

- Train Accuracy achieved: 89.552 %
- Test Accuracy acheived: 86.207 %


STEPS TO RUN FILES:


For Data Pre-Processing:

- run the src/data_preprocessing.py file
- ( If data_preprocessing.py is run, the LBW_dataset_cleaned.csv will be overwritten as the cleaned and uncleaned dataset is included in the folder )

To train and test the Neural Network:

- run the src/Neural-Net.py file
- ( Can run Neural-Net.py without running data_preprocessing.py as the cleaned and uncleaned datasets are both included in the folder ) 


src folder:
- Neural-Net.py : code to train and test the neural network model
- data_preprocessing.py : code to clean the dataset
- LBW_dataset_cleaned.csv : cleaned dataset
- LBW_dataset_uncleaned.csv : un-cleaned dataset

data folder:
- LBW_dataset_cleaned.csv : cleaned dataset
- LBW_dataset_uncleaned.csv : un-cleaned dataset





