# Airbnb_data_analysis

### 🔍 Overview

This project aims to help **Airbnb** establish a business strategy through data-driven analysis.

First, I explored what defines *high-quality accommodations*. Since guests determine whether a stay is great or not, I focused on **review scores**. By identifying which factors influence these scores, I defined the key elements that make a great listing.

Second, I conducted **NLP analysis** to pinpoint specific issues. Using topic modeling, I identified areas that need improvement and developed actionable ideas.

Third, I applied **machine learning** to optimize Airbnb listing prices based on a wide range of features.

Finally, I designed a **prototype service** to help Airbnb hosts maximize their profit.

---

### 🗂 About the Dataset

1. **Airbnb listing data in Prague, 2024 (Kaggle)**
    - **Number of features:** 75
    - **Number of instances:** 11,446
    - **Feature categories and examples:**
        - **🏠 Listing information :** name, listing_url, room_type, amenities, etc.
        - **🗺 Location information :** neighbourhood_cleansed, latitude, longitude, etc.
        - **👤 Host information :** host_name, host_since, host_location, host_is_superhost, host_response_rate, etc.
        - **💰 Booking and price information :** price, minimum_nights, maximum_nights, availability_30, etc.
        - **⭐ Review and rating information :** number_of_reviews, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value
2. **Web-scraped text data from the official Airbnb host forum**
    - Posts containing the keyword “cleaning”
    - Scrape date: September 2025
    - Number of instances: 1,000

---

### 📚 Python Libraries

- **Basic :** `os`, `datetime`
- **Web scraping :** `selenium`
- **Preprocessing & analysis :** `pandas`, `numpy`, `kagglehub`, `missingno`, `ast`, `scipy`
- **NLP :** `spacy`, `LatentDirichletAllocation`, `wordcloud`
- **ML :** `sklearn`, `xgboost`, `catboost`, `lightgbm`, `optuna`, `shap`
- **Visualization :** `matplotlib`, `seaborn`, `folium`

---

### 📊 EDA (Exploratory Data Analysis)

The goal of EDA was to identify the factors that make listings successful.

Since **review scores** directly reflect guest satisfaction, I analyzed patterns in the scoring system.

Airbnb reviews consist of one **overall rating** and six **detailed category scores**: accuracy, cleanliness, check-in, communication, location, and value.

Importantly, the overall score is **not** a simple average of the detailed scores.

Key findings:

1. The distributions of **overall**, **cleanliness**, and **value** scores were wider than others — showing more variability among listings.
   <img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/06f946f4-0368-4b52-a433-7c32da5949e2" />

2. **Private listings** had higher average scores than shared ones.
   <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/b8001ad0-9028-44d1-ba27-b72ee886ea45" />

3. Listings with **more amenities** tended to have slightly higher scores, though the effect was not strong.
   <img width="673" height="466" alt="image" src="https://github.com/user-attachments/assets/2644d957-3141-4f60-bb06-7274877204f9" />

4.  From regression analysis, **value** and **cleanliness** were the two most influential factors affecting the overall score.
   <img width="855" height="466" alt="image" src="https://github.com/user-attachments/assets/699af547-1f7c-426d-b2b1-a486256586ae" />


👉 Based on these results, I focused on **price (value)** and **cleanliness** for deeper analysis and actionable insights.

---

### 💬 NLP Analysis

To address cleanliness-related issues, I conducted text mining using 1,000 posts from the official **Airbnb Host Community Forum** containing the keyword *“cleaning.”*

Using **LDA topic modeling**, I identified two main discussion themes:

1. Disputes between hosts and guests regarding **cleaning fees**
2. Hosts **seeking recommendations** for reliable cleaning companies

    <table>
      <tr>
        <td><img src="https://github.com/user-attachments/assets/9db694f6-1260-4296-abdd-36d852c43d39" alt="image1"></td>
        <td><img src="https://github.com/user-attachments/assets/483f5da5-78fd-40ab-aa28-4f1f00e2f099" alt="image2"></td>
        <td><img src="https://github.com/user-attachments/assets/d87fea16-19f8-407a-9217-60c15ead9918" alt="image3"></td>
      </tr>
    </table>



These results suggested that many hosts struggle to find trusted cleaning services and spend excessive time managing this task.

💡 **Action item:** *“Airbnb Clean Partners”*

A **cleaning company brokerage service** where Airbnb partners with verified cleaning providers.

Hosts can simply select a preferred cleaning company on their dashboard, and Airbnb handles the rest.

This service would:

- 📈 Generate **additional revenue** for Airbnb (commission from cleaning services)
- ⏰ **Save hosts’ time** in finding cleaners
- ✨ **Improve guest satisfaction** through consistent cleanliness standards

---

### 🤖 Machine Learning

Since **price** significantly affects review scores, I built a model to help hosts set reasonable, data-driven prices.

### 🔧 ML Process

1. **Preprocessing**
    - Removed unnecessary columns
    - Filtered inactive hosts using the `availability_365` column
        - `365` → never booked (too expensive or inactive host)
        - `0` → fully booked or temporarily blocked
        - Both considered inactive and removed
2. **Derived Features**
    - **`nearby_avg_price`**: average price within 1 km (to reflect local market trends)
    - **`has_amenity`**: boolean columns indicating whether a listing includes specific amenities
        - Amenities were grouped into:
            - *Core essentials* (e.g., kitchen, Wi-Fi, heating)
            - *Nice-to-have comforts* (e.g., balcony, parking, TV)
            - *Target-specific* (e.g., crib, workspace, pet-friendly)
        - Columns with low statistical significance were excluded via t-tests.
3. **Data Splitting**
    - 80% training / 20% testing
4. **Missing Value Imputation**
    - Imputed `bathrooms`, `bedrooms`, `beds` using group means by `accommodates`
    - Imputed host-related features (acceptance, response, review scores) by `host_is_superhost`
5. **Encoding**
    - Applied one-hot encoding to categorical features (`room_type`, `neighbourhood_cleansed`, `nearest_center`)
6. **Scaling**
    - Scaled numerical features using `StandardScaler`
7. **Log Transformation**
    - Applied log transformation to `price` to reduce skewness
      
        <table>
          <tr>
            <td><img src="https://github.com/user-attachments/assets/735daff4-9dee-4f53-bac4-4be29ede8e4e" alt="image1"></td>
            <td><img src="https://github.com/user-attachments/assets/e91b21d4-5d43-4e88-b9fd-0531c11b74fc" alt="image2"></td>
          </tr>
        </table>

8. **Base Model Testing**
    - Tested 9 regression models with default parameters
    - Selected top 3 (XGBoost, CatBoost, LightGBM) based on R², RMSE, and MAE
      
      |Model|RMSE|MAE|R²|
      |------|---|---|---|
      |CatBoost|1,251.56|691.67|0.6516|
      |XGBoost|1,276.68|718.66|0.6375|
      |LightGBM|1,298.95|723.18|0.6247|
      |Random Forest|1,364.13|756.92|0.5861|
      |SVM|1,404.76|763.13|0.5611|
      |Gradient Boost|1,453.81|826.05|0.5299|
      |KNN|1,589.67	|921.52|0.4379|
      |AdaBoost|1,721.91|1,096.47|0.3405|
      |Decision Tree|1,824.22|1,085.38|0.2598|

9. **Base Model Optimization**
    - Tuned hyperparameters using **Optuna** for efficiency over grid/random search
10. **Meta Model Selection**
    - Compared Ridge, Lasso, ElasticNet, Gradient Boost, and Random Forest as meta models
    - Chose **Gradient Boost** for its strong performance with log-transformed targets
  
      |Model|RMSE|MAE|R²|
      |------|---|---|---|
      |Gradient Boost|1199.15|658.04|0.6802|
      |Ridge|1204.02|650.24|0.6776|
      |Elastic Net|1204.97|650.38|0.6770|
      |Lasso|1205.14|650.26|0.6770|
      |Random Forest|1234.69|699.61|0.6609|
  
11. **Meta model Optimization**
      - Tuned hyperparameters using Optuna for the meta model **(Gradient Boost)**
12. **Final Model**
    - Stacking Ensemble with **XGBoost**, **CatBoost**, and **LightGBM** as base models and **Gradient Boost** as the meta model
    - **Final Model Performance:**
        - RMSE: ~1190
        - MAE: ~650
        - R²: ~0.683

---

### 📈 Model Insights

1. **Feature Importance:**
    - Top 3: `availability_365`, `host_duration`, `nearby_avg_price`
    - Interpretation: booking activity, host experience, and local market influence price.
      
      <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/2504bf45-0055-4e05-a1e0-c65a4515e057" />

2. **Permutation Importance:**
    - Top 3: `accommodates`, `nearby_avg_price`, `booking_rate`
    - Interpretation: size, local market, and popularity predict price accuracy.
      
      <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/fb9925a5-fe46-4130-9cc0-08d4744157f0" />

3. **SHAP Values:**
    - Positive correlation: `accommodates`, `nearby_avg_price`
    - Negative correlation: `booking_rate`, `number_of_reviews`
    - Interpretation: larger or better-located listings cost more, while overly expensive ones tend to have lower bookings and reviews.
      
      <img width="781" height="740" alt="image" src="https://github.com/user-attachments/assets/d52a9fe3-4809-4c32-aa7c-d4e3b5b8f4ff" />
        
✅ This model helps hosts **set data-informed prices**, improving perceived *value* and *review scores*.

---

### 💡 Prototype Service

Based on the machine learning insights, I found a clear tendency that listings with certain amenities tend to have higher prices on average.
    <table>
      <tr>
        <td><img src="https://github.com/user-attachments/assets/d22f8d07-1fd6-4789-bea6-e1b4151ade92" alt="image1"></td>
        <td><img src="https://github.com/user-attachments/assets/cbd6c9be-830c-48eb-ae10-f8fb12b95b61" alt="image2"></td>
        <td><img src="https://github.com/user-attachments/assets/36ce760c-3402-4196-9234-9beeaad958c7" alt="image3"></td>
      </tr>
    </table>

Using these findings, I designed a prototype web service that provides hosts with an “insight report” — showing how small upgrades can increase their potential revenue.

Through this service, hosts can:
- Input their listing details
- Receive a data-driven recommended price
- Simulate how adding specific amenities would affect profit
- Compare price gaps with similar listings that already have those amenities

This tool helps hosts boost both revenue and listing quality through data-informed improvements.
