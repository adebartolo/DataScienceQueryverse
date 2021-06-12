# An Analysis of Employee Performance and Promotion Using Machine Learning Algorithms

### **Abstract**

The application of predictive analytics to the field of human resources management continues to evolve as organizations seek to understand how to develop and retain top talent. In this study, several employee attributes were examined to understand what company functions they worked with, acknowledge their accomplishments throughout their tenure, and analyze which performances resulted in a promotion. We found that there is not an individual attribute that results in a promotion. However, the combination of many moderately influential characteristics was found to play a role in whether or not an employee meets their key performance indicators, receives a high training score, or is promoted. We have applied three machine learning algorithms; K-Nearest Neighbor, Multiple Linear Regression, and Decision Tree, and the general implication of these findings is presented.

### **Introduction**

Human Resources (HR) is a function that manages all matters associated with employees in the workplace. Traditionally, HR manually inputs and tracks employee data using several spreadsheets. HR departments oversee data which includes employee count and performance reviews, employment type and wages, training expenses, and more. HR faces ongoing challenges related to recruitment, employee retention, and motivation. They balance these challenges while aiming to comply with regulatory policies, confidentiality agreements, and equitable treatment for all while striving for a diverse and inclusive workforce.

Making decisions that are efficient and effective can be complex; thus, leveraging analytics can enhance HR by using tools and techniques to forecast the impact of people policies on employee happiness, performance, and promotion. HR can use machine learning techniques to set and measure strategic key performance indicators involving employee performance, understand how to make unbiased decisions regarding promotions and salaries, and improve employee engagement while enhancing long-term development outcomes.

The paper is organized as follows: the dataset description and methodology are described in Section II, the setup of ML algorithms are explained in Section III, the experiment results are reported in Section IV, and concluding remarks can be found in Section V.




### **Related Work**

As noted in a publication by John M Kirimi [5], organizations rely on human capital to produce value, so it is essential to implement performance management strategies to measure and improve performance. In his paper, Kirimi explained how he used Data Mining models, which incorporate Logistic Regression, Random Forest Classifier, and K-Nearest Neighbor Classifier, to analyze extracted data from an employee appraisal form from HR departments in Kenya. Results from his study show that employee performance was highly influenced by an individual's gender, experience, academic qualification, level of training, marital status, and prior performance appraisal scores [5]. 


### **Methodology**

**Dataset**

We used the HR Analytics Case Study dataset published on Kaggle by Shivan Kumar in August of 2020. The data has 54,808 rows and 14 columns and was released by a company’s HR Department. The dataset provided information such as Employee ID, Department, Region, Education, Gender, Recruitment Channel, Number of Training, Age, Previous Year Rating, Length of Service, KPI > 80, Awards Won, Average Training Score, and Employee Promotion. The data provided background information (gender, age, education) on each employee as well as performance-based information (promotion status, KPI >80, Average Training Score). The relationship among several of these attributes is explored in this project.

**Data Pre-Processing**

Before beginning the analysis, the data were examined for outliers, noise, null values, and other characteristics that could interfere with statistical operations. Numerical values that appeared as null were changed to 0. Null values only occurred when someone had not worked there long enough to have gone through training. Normalization was used when running the KNN algorithm. All features were either qualitative and quantitative data types and were examined for correlation.

**Classifier Design**

Three different types of classifiers were chosen, K-Nearest Neighbor, Multiple Linear Regression, and Decision Tree. The models created for all three classifiers were both trained and evaluated using the same dataset provided by Kaggle. 

### **Experiments**

**K-Nearest Neighbor**

The baseline model was trained with a k = 235 and had an accuracy rating of 92.45%. Normalization was used to regulate the database as the values were vastly different and would distort the algorithm. The data that was chosen to predict promotion status were performance values and the age of the employee. The pieces of data used were Number of Training, Previous Year Rating, KPI > 80, Awards Won, and Average Training Score. This accuracy rating showed a slight improvement in comparison to a separate model that used the same data as previously listed but also included the age of the employee. The accuracy rating for this separate model was 92.20%. 

**Multiple Linear Regression**

Multiple Linear Regression was used to create several different models in an attempt to find a high accuracy rating. Of the five different models created, the model with the highest accuracy rating was model five with 39.57%. Model five predicted Length of Service using Age, KPI>80, Number of Training, Awards Won, and Promotion Status. The next highest accuracy rating is 23.7% and used all variables to predict promotion. 

**Decision Tree**

The Decision Tree algorithm was used to navigate several courses of action as it relates to an employee’s experience. The first model was constructed to predict if an employee would be promoted based on meeting performance goals, Number of Training, Previous Year Ratings, Length of Service, Awards Won, and Average Training Score. This model held an accuracy rating of 93.39%. In comparison, the second model that was created had an accuracy rating of 92.05%. 

### **Results**

Among the three algorithms, Decision Tree had a higher performance than the other algorithms. The model had an accuracy rating of 93.39%. The Decision Tree was created to predict promotion status using pieces of the dataset and proved to be the most reliable of all three algorithms. 

The next highest accuracy rating was K-Nearest Neighbor. The KNN model that was created had an accuracy rating of 92.45%. The model was created to determine if promotion status could be predicted using machine learning and the quantitative values of the dataset. Given the high accuracy rating, the model is reliable. 

Multiple Linear Regression was the least accurate algorithm with an accuracy rating of only 39.57%. The goal of the model was to predict what might contribute to an employee staying at a company for a long time. The low accuracy rating shows this model is unreliable and that the data has a low correlation. 


### **Conclusion**

The analysis presented in this paper mainly revolves around the prediction of whether or not employees receive a promotion or discovering what factors contribute to long-tenured employment. This is only one application, as machine learning is increasingly being used to uncover new insights for Human Resources. Further analysis can predict the amount of time or experience it will take until an employee is promoted to a certain level of advancement or receives a certain raise. It can also predict how long an employee will stay at a company given their track record, which can help employers understand why workers leave after a certain period, how employers should go about branding the position to attract new talent, and when to start recruiting for open positions.

There are endless possibilities with what can be accomplished by implementing machine learning algorithms. Predictive analytics empowers human resources to forecast the impact of employee training and career progression on employee performance and satisfaction. The combination of using historical data and current trends can also help companies refine standards of performance expectations, look for candidates who are a good culture fit, understand the likelihood of how long employees will remain engaged, and much more. Employees can also use this data to understand what it takes to be an exemplary worker, how long until their next promotion, and how to best proceed with their career goals. All in all, leveraging analytics sharpens business intelligence and enables companies to learn from what works or how they can improve to better support their workforce. Future models can be trained to consider an individual organization’s key performance indicators, hierarchical structure, and values to generate relevant rules for predicting employee performance.
Acknowledgment
The authors of this paper would like to thank Professor Mohammad Z Chowdhury for access to resources regarding the relevant algorithms, tools, and techniques used.

### **References**

[1] A Data-driven Analysis of Employee Promotion: The Role of the Position of Organization. 
IEEE Xplore. (n.d.). 

[2] Bart Baesens, S. D. W. (2016, December 1). Is Your Company Ready for HR Analytics? MIT Sloan Management Review. 

[3] E. Muscalu and A. Şerban, “HR Analytics for Strategic Human Resource
Management”, in Proc. 8th International Management Conference on
Management Challenges for Sustainable Development, Bucharest, Romania,
November 6th -7th, 2014, pp. 939.

[4] Harbert, T. (2020, May 12). People analytics explained. MIT Sloan. 

[5] Kirimi, J. M., & Moturi, C. A. (2016). Application of data mining 
fication in employee performance prediction. International Journal of Computer Applications, 146(7), 28-35.

[6] L. Smeyers, “7 Benefits of Predictive Retention Modeling (HR analytics).”
