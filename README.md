## Short Note about insightsR
**insightsR is an open source web application that provides automated insights for tabular data**. Insights will be statistically generated based on the target column (dependent variable) selected by the user. 

User is only expected to provide a dataset in csv format, select a numeric Target column and click 'Generate Insights' button in the left pane. Rest is taken care by the tool. Further amplifying the ease of use with a statement: **Just 2 inputs are required from the end user (1. Dataset and 2. Target column)**

## Insights provided and how to interpret

Once the user uploads the data to be analysed and provides the target column against which insights to be provided, insightsR tool will handle **pre-processing of data** which includes removal of null columns, highly correlated columns, columns that has more than 75% null values, identifies and  converts date time columns to multiple columns that helps in providing better insights. **This pre-processing step is the 1st step.**

After pre-processing, insightsR tool will statistically analyse the dataset using machine learning algorithms and provides insights below:
- Provides **top contributing features against the target data column**. For eg) if target is 'Sales' features that might turn important is price, seasonality etc.  insightsR tool provides such highly important features and % of contribution.
- Provides **data visualizations for top 5 contributing features**. insightsR tool puts numeric features into multiple buckets to provide a different perception by aligning data into bucketed categories.
- Moving on, deeper insights are provided based on identified top contributors. Based on a sample picked from the top contributors, insightsR first provides **info on whether the contribution is +ve or -ve and by what %**. Features that brought down/up the mean target value is an immensely useful insight. Further within the sample how did each feature contribute is detailed.
- Finally, the tool **simulates a scenario:** what happens when the contributor data stays constant and every other column data remains as is - will the target value improves/degrades? This data will provide us view on whether the contributing features has really impacted the target value or is it just part of larger multiple contributing features combined.

Based on the insights provided, user could identify areas that can be improved, devise a plan on what would happen if a change is simulated, devise a plan to achieve the target.


## Installation
Pre-requisite: Python(~3.7.9) is installed in the system.
Execute the following commands in Terminal/cmd 

> git clone https://github.com/Vinothsuku/insightsR.git

> cd .\insightsr

> pip install -r requirements.txt

> streamlit run insightsr.py

Open a browser and goto: http://localhost:8501

(Optional - Suggested) Create a virtual environment first and then execute the above mentioned commands.

## Blog
I have written a detailed blog [here](https://medium.com/analytics-vidhya/insightsr-automated-insights-for-tabular-data-8328a67de3ed).


## Try out online
Hosted [online](https://insightsr.herokuapp.com/). Feel free to try out with any dataset in csv format.


## Constraints
Currently the tool supports only datasets in "csv" and provides insights for a target column that has continuous values (regression type) for eg) total sales, price, cost, income, salary etc.


## Credits
[fast.ai](https://fast.ai) is the base for this tool. I have put it together as an automation effort (kind of basic autoML type) and as a web application that could be used by anyone with ease.
