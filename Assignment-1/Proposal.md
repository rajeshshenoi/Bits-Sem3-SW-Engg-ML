# Proposal: Customer Churn Prediction System for Telecom Industry

## Group Details
**Group No:** Group-1  
**Group Members:**
- Member 1 - BITS ID: [Insert BITS ID]
- Member 2 - BITS ID: [Insert BITS ID]
- Member 3 - BITS ID: [Insert BITS ID]
- Member 4 - BITS ID: [Insert BITS ID]

---

## 1. Problem Statement

**Domain:** Telecommunications Industry  
**Application:** Customer Churn Prediction System

**Problem:** Customer churn (attrition) is a critical issue for telecommunications companies. Losing existing customers is significantly more expensive than retaining them. Current manual approaches to churn prediction are often reactive and lack the ability to proactively identify customers at risk of leaving.

**Objective:** Develop a machine learning-based system that predicts the likelihood of a customer churning within the next 30 days, enabling proactive retention strategies.

---

## 2. Requirements Formulation

### 2.1 Functional Requirements

| ID | Requirement Description | Priority |
|----|-------------------------|----------|
| FR1 | System shall accept customer demographics and usage data as input | High |
| FR2 | System shall predict churn probability for each customer | High |
| FR3 | System shall provide churn risk classification (Low/Medium/High) | High |
| FR4 | System shall generate actionable insights and recommendations | Medium |
| FR5 | System shall support batch processing for multiple customers | Medium |
| FR6 | System shall log all predictions with timestamps | Low |

### 2.2 Non-Functional Requirements

| ID | Requirement Description | Priority |
|----|-------------------------|----------|
| NFR1 | System shall process predictions within 2 seconds per batch | High |
| NFR2 | System shall maintain >90% prediction accuracy | High |
| NFR3 | System shall support up to 10,000 customers per batch | Medium |
| NFR4 | System shall be deployable as a REST API | Medium |
| NFR5 | System logs shall be audit-traceable | Low |

### 2.3 Measurable Goals

1. **Accuracy Goal:** Achieve at least 85% F1-score on the test dataset
2. **Latency Goal:** Predict churn probability for 1000 customers within 10 seconds
3. **Adoption Goal:** Enable business users to access predictions via a web dashboard
4. **ROI Goal:** Reduce churn rate by 15% through proactive interventions

---

## 3. GR4ML Notations

### 3.1 Business View

```
[Business Goal]
├── Increase Customer Retention Rate
│   └── [Business Metric] Retention Rate (%)
│       └── Target: Increase by 15% in 6 months
│
└── Reduce Churn Rate
    └── [Business Metric] Churn Rate (%)
        └── Target: Reduce from 12% to 8% in 6 months
```

**Stakeholders:**
- **Telecom Company Management:** Decision-makers for retention strategies
- **Customer Service Team:** Frontline staff implementing retention actions
- **Marketing Team:** Designing personalized offers for at-risk customers
- **Data Science Team:** Developing and maintaining the ML model

### 3.2 Analytics Design View

```
[Input Data]
├── Customer Demographics (age, gender, tenure)
├── Usage Patterns (calls, data usage, SMS count)
├── Billing Information (monthly charges, total charges)
└── Service Information (contract type, payment method)
      |
      v
[Feature Engineering]
├── Feature Selection
│   └── Select most predictive features using correlation analysis
│
├── Feature Transformation
│   ├── Normalize numerical features (StandardScaler)
│   └── Encode categorical features (OneHotEncoder)
│
└── Feature Importance Analysis
    └── Use Random Forest to identify top predictive features
      |
      v
[ML Model]
├── Model Selection: Random Forest Classifier
│   └── Chosen for: Interpretability, handles mixed data types
│
├── Training Strategy
│   ├── Training Set: 70%
│   ├── Validation Set: 15%
│   └── Test Set: 15%
│
└── Evaluation Metrics
    ├── Accuracy: >85%
    ├── Precision: >80% (minimize false positives)
    ├── Recall: >75% (minimize false negatives)
    └── F1-Score: >78%
      |
      v
[Output]
└── Churn Prediction + Probability Score
    ├── Low Risk (<30%)
    ├── Medium Risk (30-70%)
    └── High Risk (>70%)
```

### 3.3 Data Preparation View

```
[Data Sources]
├── Primary: Telecom Customer Database
│   ├── Table: customers
│   │   ├── customer_id (PK)
│   │   ├── gender
│   │   ├── senior_citizen
│   │   ├── partner
│   │   └── dependents
│   │
│   ├── Table: contracts
│   │   ├── contract_id (PK)
│   │   ├── customer_id (FK)
│   │   ├── contract_type
│   │   ├── tenure_months
│   │   └── monthly_charges
│   │
│   ├── Table: services
│   │   ├── service_id (PK)
│   │   ├── customer_id (FK)
│   │   ├── internet_service
│   │   ├── online_security
│   │   └── tech_support
│   │
│   └── Table: churn_history
│       ├── churn_id (PK)
│       ├── customer_id (FK)
│       ├── churn_date
│       └── churn_flag
│
└── Secondary: External Data (optional)
    └── Economic Indicators
        ├── unemployment_rate
        └── inflation_rate
      |
      v
[Data Pipeline]
├── Data Ingestion
│   └── Query telecom database for customer records
│
├── Data Cleaning
│   ├── Handle missing values (impute with median/mode)
│   ├── Remove duplicates
│   └── Fix data type inconsistencies
│
├── Feature Engineering
│   ├── Create new features:
│   │   ├── tenure_category (short-term, mid-term, long-term)
│   │   ├── charge_ratio (monthly_charges / total_charges)
│   │   └── service_count (number of add-on services)
│   │
│   └── Encode categorical variables
│
└── Data Validation
    ├── Check data quality metrics
    └── Log data transformation history
      |
      v
[Feature Store]
└── Processed features ready for model inference
```

---

## 4. Top 3 Quality Requirements

### 1. **Accuracy and Predictive Performance**
**Justification:** The primary purpose of the system is to accurately identify customers at risk of churning. Low accuracy would lead to wasted retention resources on customers who wouldn't churn anyway (false positives) or missing customers who are actually at risk (false negatives). For a churn prediction system, both precision and recall are critical - we want to identify most at-risk customers while not wasting resources on those who won't churn.

**Metrics:** F1-Score > 85%, Recall > 75% (to minimize missed churn cases)

### 2. **Scalability and Performance**
**Justification:** The system must handle varying workloads - from real-time single predictions to batch processing of millions of customers. The telecom company may have peak times (e.g., end of month billing cycles) where many predictions are needed simultaneously. The system should scale horizontally and process predictions quickly enough to be actionable.

**Metrics:** Support 10,000 customers/batch, <2 seconds processing time

### 3. **Interpretability and Explainability**
**Justification:** Business stakeholders need to understand why a customer is flagged as high-risk to implement effective retention strategies. If the model is a "black box" that can't explain its predictions, the insights won't be actionable. Feature importance and local explanations (SHAP values) will help the marketing and customer service teams understand what factors drive churn for each customer.

**Implementation:** Use Random Forest (inherently interpretable) with SHAP for local explanations

---

## 5. System Architecture

```
+-------------------+
|   User Interface  |
| (Web Dashboard /  |
|  API Consumers)   |
+-------------------+
         |
         | REST API / Web Interface
         v
+-------------------+       +------------------+
|   API Gateway     |<----->|   Load Balancer  |
|  (FastAPI/Flask)  |       +------------------+
+-------------------+
         |
         v
+-------------------+
|   Application     |
|   Layer (Python)  |
+-------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +------------------+
|Model  |  |   Feature Store  |
|Server |  |  (Pre-computed   |
|(MLflow)|  |   features)      |
+-------+  +------------------+
    |
    v
+-------------------+
|   Data Sources    |
| (PostgreSQL/MySQL)|
+-------------------+

+-------------------+
|   Data Pipeline   |
| (ETL / DAG)       |
+-------------------+
    |
    v
+-------------------+
|   Feature Store   |
| (Redis/PostgreSQL)|
+-------------------+
```

### Components:

1. **User Interface:** Web dashboard for business users and REST API for programmatic access
2. **API Gateway:** Handles authentication, rate limiting, and request routing
3. **Application Layer:** Main application logic in Python using FastAPI
4. **Model Server:** ML model serving using MLflow
5. **Feature Store:** Stores pre-computed features for quick inference
6. **Data Sources:** Relational database storing customer and billing data
7. **Data Pipeline:** Scheduled ETL jobs for data preprocessing and feature engineering

---

## 6. Architectural Patterns

### Pattern 1: **Pipeline Pattern**
**Application:** Data preparation and model inference pipeline

**Description:** The system uses a pipeline architecture where data flows through sequential stages:
1. Data Ingestion → 2. Data Cleaning → 3. Feature Engineering → 4. Model Prediction → 5. Result Storage

**Benefits:**
- Clear separation of concerns
- Easy to test each stage independently
- Allows for parallel processing of batches
- Simple to modify or replace individual stages

**Implementation:**
- Python pandas for data transformation
- scikit-learn for preprocessing and model
- Airflow for pipeline orchestration

### Pattern 2: **Microservices Pattern**
**Application:** Separation of core services

**Description:** The application is decomposed into independent services:
- **API Service:** REST endpoints for predictions
- **Model Service:** ML model loading and inference
- **Data Service:** Database operations and feature store access
- **Pipeline Service:** Scheduled data processing jobs

**Benefits:**
- Independent deployment and scaling
- Fault isolation (failure in one service doesn't affect others)
- Technology flexibility (each service can use appropriate tech stack)
- Easier team organization

**Implementation:**
- FastAPI for API service
- MLflow for model serving
- PostgreSQL for data persistence
- Celery for background jobs

---

## 7. Implementation Plan

### Week 1-2: Requirements and Design
- Finalize problem statement and requirements
- Complete GR4ML notations and architecture diagrams
- Set up development environment

### Week 3-4: Data Pipeline Implementation
- Connect to data sources
- Implement data cleaning and feature engineering
- Create feature store
- Set up ETL pipeline

### Week 5-6: Model Development
- Exploratory data analysis
- Train and evaluate ML models
- Select best model and optimize
- Implement model versioning with MLflow

### Week 7-8: System Architecture
- Implement API service (FastAPI)
- Integrate with model server
- Set up feature store access
- Create data service layer

### Week 9-10: Testing and Documentation
- Unit and integration testing
- Performance testing
- Create user documentation
- Write final report

### Technologies:
- **Programming:** Python 3.10+
- **ML Framework:** scikit-learn, XGBoost
- **API Framework:** FastAPI
- **Model Management:** MLflow
- **Database:** PostgreSQL
- **Pipeline:** Apache Airflow / Custom Python scripts
- **Containerization:** Docker (optional for deployment)

---

## 8. Submission Deliverables

1. **Proposal Document** (this file) - `proposal.md`
2. **Implementation Notebook** - `Group1.ipynb` containing:
   - Data exploration and preprocessing
   - Model development and evaluation
   - System architecture implementation
   - API implementation
3. **Final Report** - `Group1.pdf` with:
   - Complete system documentation
   - Screenshots of implementation
   - Performance metrics
   - Member contributions

---

**Status:** Proposal Ready for Review  
**Next Steps:** Begin implementation phase with data pipeline development