 House Price Prediction

This project aims to build a machine learning model to predict house prices based on various features such as location, size, number of bedrooms, and other relevant attributes. The goal is to provide accurate price predictions to aid real estate investors, buyers, and sellers in making informed decisions.

System Requirements

- Hardware: 8GB RAM, 10GB storage, multi-core processor.
- Software; Python 3.8+, Windows/macOS/Linux.

Libraries Required

- Data Processing: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`
- Deployment: `flask`, `joblib`

Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
```

 Usage
 
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Run data preprocessing and model training scripts:
   ```bash
   python preprocess.py
   python train_model.py
   ```
4. Start the Flask server:
   ```bash
   python app.py
   ```
5. Access the web application at `http://localhost:5000` to input house features and get price predictions.

Conclusion :

The project successfully developed a model using Gradient Boosting Regressor, which provides accurate house price predictions.
This model helps real estate stakeholders make informed decisions based on reliable price estimates.
