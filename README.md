# Yogamrit
An application for Yoga Guidance and Practice.

A platform for user to learn yoga with AR-based camera. It suggests the correct posture with automatic posture detection and correction. It automates pose detection and correction with deep learning and pose analysis algorithms. Primarily uses MediaPipe for pose estimation, followed by trained DL network for pose classification. The demonstrated pose is then compared with a reference to accurately correct the overall pose.

Comparison of popular multi-class classification models – DNN, CNN, LSTM, Logistic Regression, Random Forest and SGD.

<img width="250" height="250" alt="VrikAsana" src="https://github.com/user-attachments/assets/8b6f9e6f-9a06-4a18-a1a0-aa14f154c8da" />
<img width="250" height="250" alt="Tadasana" src="https://github.com/user-attachments/assets/c87b6724-64ab-4657-9f8a-98ad10da5fce" />
<img width="250" height="250" alt="Vajrasana" src="https://github.com/user-attachments/assets/a4a6de01-3538-4184-8366-1e50e1bf02f7" />

<img width="250" height="250" alt="Padmasana" src="https://github.com/user-attachments/assets/653431e3-b76d-4d35-ab1d-24e13ff128aa" />
<img width="250" height="250" alt="Panchim Uttanasan" src="https://github.com/user-attachments/assets/3e5bf8e4-d104-4785-9ff1-042c150cc9c0" />
<img width="250" height="250" alt="Bhujangasana" src="https://github.com/user-attachments/assets/20d935dd-55c0-41e2-9f2a-090e69a3d2e5" />

***Faces are blurred for privacy reasons!***

A detailed blog and conference paper is published, and this repository showcases its implementation.

- Detailed blog: [Medium Blog](https://medium.com/@samyakwaghdhare/revolutionizing-yoga-how-ai-powered-auto-detection-and-correction-is-transforming-your-practice-d516479022b8)
- Conference paper: [IEEE Paper](https://doi.org/10.1109/ICRITO66076.2025.11241748)

## Related Notebooks
Model training and results: [Kaggle Notebook](https://www.kaggle.com/code/samyak03/yogamrit-models)

## Installation
The Python version used for this project is **3.12.10**. It can be said that the libraries used with the current version will be compatible with *3.12.x* version of Python.

### Clone the repository
Clone this github repository.
```sh
git clone https://github.com/SAM-DEV007/MonoVision.git
cd MonoVision
```

If Git is not installed, the repository can be cloned or downloaded by clicking on Code drop-down menu and selecting either Open with Github Desktop (requires Github Desktop installation in the local machine) or Download Zip to download the repository contents.

### Virtual Environment
Create and activate the virtual environment.

#### Windows
```sh
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/Mac
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
The virtual environment should be activated. Ensure that you are in the home directory of the repository or it may fail to find `requirements.txt`.

```sh
pip install -r requirements.txt
# or 
pip3 install -r requirements.txt
```

## Dataset
The raw dataset used is private from a certified yoga practitioner. `Data` folder contains the numerical data extracted from the video frames of the raw dataset. For own dataset, it is recommended to take videos of posing the yoga poses from varying angles.
- `Dataset_Config/pose_visualize.py` - Create and save pose correction references.
- `Dataset_Config/video_csv.py` - Create and save numerical pose data from videos for model training.

## Important Files
- `pose_video.py` - Pose detection and correction using OpenCV native window.
- `app.py` - Pose detection and correction using flask web application.
- `model_train.py` - Train the model with the pose dataset.

## Statistical Comparison
### Model Comparison
![Tabular Comparison](https://github.com/user-attachments/assets/ff5d68be-7eda-4bd7-9806-fea044f24fa3)

![Bar Plot Comparison](https://github.com/user-attachments/assets/9e8a57bb-85c9-4ff1-9c30-34672bc07f9a)

![Time Comparison](https://github.com/user-attachments/assets/21dd9b58-ab07-4c77-9c6a-f8b7d6affdbc)

### Statistical Significance
Pairwise comparisons between models were performed using the two-sided Mann–Whitney U test. This non-parametric test was used to assess whether the distributions of each evaluation metric differed between pairs of models without assuming normally distributed observations.

For six models, all possible pairwise comparisons were performed, resulting in 15 comparisons per metric. Because multiple statistical tests were performed, the resulting p-values were adjusted using the Benjamini–Hochberg False Discovery Rate (FDR) procedure.

![Statistical Significance](https://github.com/user-attachments/assets/cd255d2b-e106-4433-98c5-50ec6708cb54)

**Pairwise comparisons for accuracy using Mann-Whitney U test with false discovery rate (FDR) for statistical significance. ***(In the output displaying the standard deviations, `*` means that the null hypothesis is rejected. The plot actually shows the degree of statistical significance.)*****

|  Corrected p-value |   Star (In Plot)  | Interpretation                   |
| -----------------: | :-----: | -------------------------------- |
|        `p < 0.001` |  `***`  | Highly statistically significant |
| `0.001 ≤ p < 0.01` |   `**`  | Statistically significant        |
|  `0.01 ≤ p < 0.05` |   `*`   | Statistically significant        |
|         `p ≥ 0.05` | No star | Not statistically significant    |

The brackets on the plots identify the pair of models being compared, while the stars indicate the statistical significance of the corresponding FDR-adjusted Mann–Whitney U test.

For example, DNN vs CNN: `**` indicates that the Accuracy distributions of DNN and CNN differed significantly according to the two-sided Mann–Whitney U test, with an FDR-adjusted p-value below 0.01.

A lack of statistical significance (`p ≥ 0.05`) indicates that the test did not provide sufficient evidence to conclude that the two model distributions differ; it does not necessarily prove that the models are identical.

## Contribution
1. Samyak Waghdhare
2. Giridhar Bargaley
3. Shrishti Singh
