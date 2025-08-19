class PromptsForSpecificFunctionality:
    merge_query_and_history = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

    answer_noncoding_query = """You are highly proficient with Python and the pandas library. A user has submitted a query that you need to address: '{input}'. 
You also have a list of subtasks that need to be completed. Normally, Your task would be to define a Python function named `def solve(df):` that fulfills the user's query and returns the result of the analysis.
    
However, this time the user does not want to generate and execute some code but he/she wants something else.
Your task is now to try to fullfill the user's wish or to answer the question as best you can.
Given the following conversation and a follow up question as below:

Chat History:
{chat_history}
Follow Up Input: {question}

Specifically, pay attention especially to the all the details in the conversation, what were the previous (analytical) steps (user queries,
e.g. "select top 5 rows", "plot the data", "group by column X", etc.) and what was the user's last query. These analytical steps may be important
for summaries, explanations, or other types of answers.

Other types of questions are general questions, when the history is not important, and the user asks for a general information or explanation.

- Your answers should be easy to understand, clear, and concise. Do not generate one information multiple times with different wording.
- You are talking to the user, so your should refer to the user as "you" and to yourself as "I".

Do, what the user wants as best as you can or answer the query as best as you can:
    """

    save_df = """ \nIf and only if the question requires pd.DataFrame type as the answer (the answer is not a string, number etc.), save the result into 'df'.
"""

    is_coding_needed_cls = """You are highly proficient with Python and the pandas library. A user has submitted a query that you need to address. 
    Your task now is to determine whether the user's query requires code to be executed in order to be answered or the answer can be given without executing any code.
    
    You have access to tabular data in the form of pandas.DataFrame named 'df'. The df looks like this: {df}

    Here are several examples of queries that require code to be executed (in the form query: class; where class can be 0 or 1):

    select top 5 rows: 0
    Compute correlations of all numeric columns with mileage_end values.: 0 (i.e. code is required)
    Select rows where the car type is any Taycan model or energy_pos is zero: 0
    Count rows for each car type with first formatted payload after January 2020: 0
    Identify days with an abnormal temperature_ambient_max that is bigger than 99 percent of the values: 0
    Compute correlations of all numeric columns with mileage_end values.: 0
    Select rows where the car type is any Taycan model or energy_pos is zero.: 0
    How many rows and columns are in the dataset?:0
    Calculate the mean and standard deviation of the battery capacity.:0
    Filter out rows where the speed is greater than 100 km/h: 0
    Generate a histogram of the temperature_ambient_min values.: 0

    How many rows and columns are in the dataset?: 0make a summary of all the steps before: 1
    write a text report based on the previous steps: 1
    how are you: 1
    describe the data in a natural language: 1
    I need to find outliers, which methods can be used?: 1
    how to normalize these data for NN trainig?: 1
    What is a SoC?: 1   #(soc - name of column)
        
    Given the query, predict the class (0/1) - output just the class number, nothing else!!
    
    Query: {input}
    Class:
    """