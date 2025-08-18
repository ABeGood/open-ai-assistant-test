import pandas as pd
from .plotting_guidelines import plotting_best_practices, seaborn_implementation
from .base_prompt_strategy import BasePromptStrategy

class PromptsSimple(BasePromptStrategy):
    has_planner = True
    max_debug_times = 2

    generate_steps_for_plot_save = """
You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{{input}}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, such as names of dataframe columns. Make sure to state saving the plot to '{{plotname}}' in the last step. Use Seaborn for enhanced visualization practices.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. Here's the beginning of the 'df': 
'{{df_head}}'

'{{functions_description}}'

Present your algorithm in up to six simple, clear English steps. Remember to explain steps rather than to write code. Implement Seaborn for plotting to adhere to the following enhanced plotting best practices:
{seaborn_implementation}

You must output only these steps, the code generation assistant is going to follow them.
In case of multiple plots, store them to separate files plot00.png, plot01.png, etc.
Think very carefully about what of plot you should use:
Pairwise data - plot, scatter, stem, bar, stairs, ..
Statistical distributions - hist, boxplot, errorbar, violinplot, pie, hist2d, ..
3D and volumetric data - scatter, plot_surface, plot_trisurf, plot_wireframe
Gridded data - contour, contourf, pcolor, pcolormesh, imshow, matshow, spy, ..
Irregularly gridded data - tripcolor, triplot, trisurf, tricontour, tricontourf, ..

Here's an example of output for your inspiration:
1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
2. Extract the 5 rows from the sorted DataFrame.
3. Multiply each found value in the extracted dataframe by 3.
4. Use Seaborn to create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Save the plot to 'plots/example_plot00.png', ensuring all elements are clear and accessible according to Seaborn's guidelines.
""".format(seaborn_implementation=seaborn_implementation)
    
    generate_steps_for_plot_show = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{{input}}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, such as names of dataframe columns. Make sure to state showing the plot in the last step using Seaborn to enhance visualization.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame `df`. 
The DataFrame `df` is already defined and populated with necessary data. Here's the beginning of the 'df': 
'{{df_head}}'

'{{functions_description}}'

Present your algorithm in up to six simple, clear English steps. Remember to explain steps rather than to write code. Ensure the plot adheres to Seaborn's enhanced plotting best practices:
{seaborn_implementation}

You must output only these steps, the code generation assistant is going to follow them.
In case of multiple plots, store them to separate files plot00.png, plot01.png, etc.
Think very carefully about what of plot you should use:

Here's an example of output for your inspiration:
1. Sort the DataFrame `df` in descending order based on the 'happiness_index' column.
2. Extract the 5 rows from the sorted DataFrame.
3. Multiply each found value in the extracted dataframe by 3.
4. Use Seaborn to create a bar plot with 'Voltage' on the x axis and multiplied 'Speed' on the y axis.
5. Show the plot, ensuring all visualization elements meet Seaborn's standards for clarity and accessibility.
""".format(seaborn_implementation=seaborn_implementation)

    generate_steps_no_plot = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Formulate your response as an algorithm, breaking the solution into steps, including any values necessary to answer the question, 
such as names of dataframe columns.

Beware of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. sorting can be dangerous!, 
try to use other build in functions than sort). Lists: if one wants a list of specific types, these tupes must be unique.

This algorithm will be later used to write a Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beggining of the 'df': 
{df_head}

{functions_description}

Present your algorithm with at most six simple, clear English steps. 
Remember to explain steps rather than to write code.
Don't include any visualization steps like plots or charts.
You must output only these steps, the code generation assistant is going to follow them. 

Here's an example of output for your inspiration:
1. Find and store the minimal value in the 'Speed' column.
2. Find and store the maximal value in the 'Voltage' column.
3. Subtract the minimal speed from the maximal voltage.
4. Raise the result to the third power.
5. Print the result.
"""

    reformulate_plan = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

We have an algorithm (plan) that will later be used to write Python code and applied to the existing pandas DataFrame 'df'. 
The DataFrame 'df' is already defined and populated with necessary data. So there is no need to define it again or load it. Here's the beginning of the 'df': 
{df_head}
{column_description}

{functions_description}

Make sure, the plan is really precise and correct. If yes, output just the plan. Else modify the plan. Especially pay attention to whether the plan
will return the correct result, whether all steps are necessary and whether the estimated value is as expected from the user query. Beware
of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. sorting can be dangerous!, 
try to use other build in functions than sort).

The plan: 
{plan}

Tuned plan:
"""


    # TODO: {input} to ''
    generate_code = """The user provided a query that you need to help achieving: {input}. 
You also have a list of subtasks to be accomplished using Python.

Beware of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. it is not safe to use sort sometimes).

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

df is already created

Example of the output format:
```python

```"""

    generate_code_multiple_dfs = """The user provided a query that you need to help achieving: {input}. 
You also have a list of subtasks to be accomplished using Python.

Beware of values order - e.g. sort might change the order, hence two rows with equeal values might have different order after sort (i.e. it is not safe to use sort sometimes).

You have been presented with several pandas DataFrames named `df_k`, where k is a number; e.g. `{df_examples}` etc.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Here are the first 5 rows of the given DataFrames, sorted (i.e. first 5 rows of `df_1`, then `df_2`, etc.):
{data}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.
Use only the provided DataFrames {df_examples} in your code.
The DataFrames `df_k` have already been defined and populated with the required data, so don't load it and don't create new ones.
Note again, there is no 'df' DataFrame, only `df_k`, where k is a number.

Example of the output format:
```python

```
"""

    # TODO: {input} to ''
    generate_code_for_plot_save = """The user provided a query that you need to help achieving: {input}. 
You also have a list of subtasks to be accomplished using Python.

You have been presented with a pandas dataframe named `df`.
The DataFrame `df` has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}

Return only the python code that accomplishes the following tasks:
{plan}

{functions_description}



Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must not include `plt.show()`. Just save the plot the way it is stated in the tasks.
You must include print statements to output the final result of your code.
In case of multiple plots, store them to separate files plot00.png, plot01.png, etc.; Each plot must have a headline to indicate what the plot means.
You must use the backticks to enclose the code.

Example of the output format:
```python

```"""

    # TODO: {input} to ''
    fix_code = """You are a helpful assistant that corrects the python code that resulted in an error and returns the corrected code.

The code was designed to achieve this user request: {input}.
The DataFrame `df`, that we are working with has already been defined and populated with the required data, so don't load it and don't create a new one.
The result of `print(df.head({head_number}))` is:
{df_head}

The execution of the following code that was provided in the previous step resulted in an error:
```python
{code}
```

The error message is: '{error}'

{functions_description}

Return a corrected python code that fixes the error.
Always include the import statements at the top of the code, and comments and print statements where necessary.
Use the same format with backticks. Example of the output format:
```python

```"""

    query_rewrite = """You are an AI data analyst and your job is to assist the user with simple data analysis.
The user asked the following question: '{input}'.

Reformulate the given question into a query (in natural language) s.t.: the query is not too wordy and more precise. It must be clear from the query,
what analytical steps to do, hence the query should be easily tranlastable to SQL or other similar language. It is extremely crucial, that the
query must ask exactly the same thing as the original question.

Just for info, the question is intended for the following table defined in pandas: 
{df_head}
{column_description}

{functions_description}

Do not change anything when not necessary.

"""

    is_query_clear = """
Your task is to help the AI analyst tool determine whether the user's query is clear or if additional information is needed. For example, if a user asks a question about voltage but there are three types of voltages (1, 2, 3), you need to know which specific voltage is meant. If you are more or less confident about the query, answer "Yes". Otherwise, reply "No, | clarifying_query: 'your_query'". You must keep the notation!

Examples:

Example 1:
ChargingCycleId,CarType,SoC_Start (pc),SoC_End (pc),Temperature (C),Current (A)
1,Taycan,21,24,12,160
2,Taycan,26,52,22,10
3,Taycan,0,8,-9,232
query: what is the highest temperature?
answer: Yes

Example 2:
ChargingCycleId,CarType,SoC_Start (pc),SoC_End (pc),Temperature (C),Current (A)
1,Taycan,21,24,12,160
2,Taycan,26,52,22,10
3,Taycan,0,8,-9,232
query: what is the highest SoC?
answer: No, | clarifying_query: Do you mean SoC_Start or SoC_End?

Example 3:
index,distance,mileage_start,mileage_end,speed_mean,speed_abs_diff_mean,speed_max,voltage_mean,voltage_first,voltage_last,payload_ts_first,payload_ts_last,soc_start,soc_end,soc_diff,current_abs_diff_mean,current_mean,current_abs_mean,car_type,records_length,power_kw_abs_mean,power_kw_mean,power_kw_median,power_kw_max,power_kw_min,energy_neg,energy_pos,temperature_ambient_first,temperature_ambient_last,temperature_ambient_min,temperature_ambient_max,temperature_ambient_mean,temperature_start,temperature_end,acceleration_neg_mean,acceleration_pos_mean,soc_on_km,data_source,sampling_rate,country_code_alpha2,timezone,payload_ts_first_formatted,payload_ts_last_formatted,year_month,str_ts_first,str_ts_last,weekday,perc_missing_data,vin
0,3.0,23483.0,23486.31047781128,65.69223445442073,14.567049652667947,82.59740423578664,809.7003595778169,795.0043656227721,719.3353641427431,1674420711276,1682247127450,70.10529833324077,56.10890525860751,20.68731732271334,16.063066679617013,-31.473484327601923,23.757686644370697,Taycan33,23,15.08699561467924,-12.02608751671494,-16.706308668622643,0.6563081459552897,-67.01769102402076,-12.178321633633074,0.0010194595535583,7.458932692957482,9.101118877040903,17.216099296549388,7.643567408807364,8.87619815576264,9.03155771028374,9.43906646374834,-0.1179969381026706,0.1422525936555543,0.3263216922582791,ZDS_ucvcol,60,PE,Europe/Brussels,2023-03-06 14:25:52,2023-01-31 11:39:05,202302,2022-11-03 14:37:12,2023-05-24 09:48:34,Saturday,0.0,2290
query: what is the average ambient temperature in the data?
answer: No, | clarifying_query: which ambient temperature, first or last? (note: it could also be ambient_max_temperature etc., we just know it must be ambient temperature)

Example 4:
index,distance,mileage_start,mileage_end,speed_mean,speed_abs_diff_mean,speed_max,voltage_mean,voltage_first,voltage_last,payload_ts_first,payload_ts_last,soc_start,soc_end,soc_diff,current_abs_diff_mean,current_mean,current_abs_mean,car_type,records_length,power_kw_abs_mean,power_kw_mean,power_kw_median,power_kw_max,power_kw_min,energy_neg,energy_pos,temperature_ambient_first,temperature_ambient_last,temperature_ambient_min,temperature_ambient_max,temperature_ambient_mean,temperature_start,temperature_end,acceleration_neg_mean,acceleration_pos_mean,soc_on_km,data_source,sampling_rate,country_code_alpha2,timezone,payload_ts_first_formatted,payload_ts_last_formatted,year_month,str_ts_first,str_ts_last,weekday,perc_missing_data,vin
0,3.0,23483.0,23486.31047781128,65.69223445442073,14.567049652667947,82.59740423578664,809.7003595778169,795.0043656227721,719.3353641427431,1674420711276,1682247127450,70.10529833324077,56.10890525860751,20.68731732271334,16.063066679617013,-31.473484327601923,23.757686644370697,Taycan33,23,15.08699561467924,-12.02608751671494,-16.706308668622643,0.6563081459552897,-67.01769102402076,-12.178321633633074,0.0010194595535583,7.458932692957482,9.101118877040903,17.216099296549388,7.643567408807364,8.87619815576264,9.03155771028374,9.43906646374834,-0.1179969381026706,0.1422525936555543,0.3263216922582791,ZDS_ucvcol,60,PE,Europe/Brussels,2023-03-06 14:25:52,2023-01-31 11:39:05,202302,2022-11-03 14:37:12,2023-05-24 09:48:34,Saturday,0.0,2290
query: Make a subtable of index, weekday, vin, mean start and last voltages and all SoCs. Include also all temperatures.
answer: Yes | (clarification: the requested columns index, weekday, vin, start_voltages, last_voltage - they are in the table. The user requested all SoCs columns (hence SoC_start, SoC_end, this is also alright). The same as for SoCs applies to temperatures.)

Example 5:
data - same as before
query: In how many driving cycles is the ambient temperature lower than the temperature acquired from averaging out all the probes at the start of the driving cycle?
answer: Yes | (clarification: the user asked for the ambient temperature, he also asked for the temperature at the start of the driving cycle, so we know that the user meant the temperature_ambient_first column)

Example 6:
data - same as before
query: Give me the median of average powers of driving cycles in megawatts measurement unit
answer: Yes, | clarifying_query: User asked for median aggregation function (the aggregation function is always before column name), he asked for average powers, so we know he meant power_kw_mean column. He asked for the power in megawatts, but kW can be easily converted to MW.

Example 7:
data - same as before
query: compute delta soc column between start and end
answer: Yes | (clarification: delta is the difference between SoC_end and SoC_start)

_______________________________________________________________

Be precise, but not strict regarding clarity. Try to find the correct columns before asking for additional information. If you are not 100% sure but one answer is rather likely, do not ask for clarification.

Check carefully the column descriptions. It may be clear from the descriptions which columns are meant. Do not ask stupid questions that are rather clear from the context.

Never ask 'Do you mean column X?', just assume that X is correct and output 'Yes' instead! E.g. clarifying_query: Do you mean power_kw_max or power_kw_mean? -> answer: Yes

Always assume that the query is clear and try to find out about potential ambiguities in the query. If the query is clearly unclear, then ask for clarification, otherwise don't.

Never ask Yes/No questions - if you want to ask for clarification, such that the answer is Yes/No, then assume the answer is Yes and answer: 'Yes'.

Determiners matter: 'all <column_name>' means all columns with the name <column_name>, 'the <column_name>' means only one column with the name <column_name> (i.e. the column name must be clearly specified). If there are more columns with the same asked name and the name is precided by 'the', then ask for clarification.
I.e. 'the + not_very_clear_column_name' -> say no and ask for clarification, 'all + not_very_clear_column_name' -> say yes and do not ask for clarification.

Queries of type calculate/compute/... generally have this scheme: VB + aggregation function + column name + optional condition. I.e. if such a query has only one name of aggregation function, it is not part of the column name (e.g. compute mean of mean temperature: VB=compute, agg=mean, column=temperature_mean versus compute mean temperature: VB=compute, agg=mean, column=temperature). In the latter case, if there are multiple columns with temperature, it is not clear which one is meant. In the first case, it is clear that the user meant temperature_mean column.

Do not ask for information that is already in the column descriptions (see them below). Do not ask for clarification about questions that are probably clear.

General rule: when you are not certain the query is missing key information, answer 'Yes'. If you cannot decide between two columns, read carefully the column descriptions and try to find out which one is more likely.

Generally - better to say yes than ask for clarification.

Our table has these columns (list of column descriptions), consider the descriptions carefully before asking for clarification.:
{column_description}

Your Task:
{df_head}
query: '{input}'
answer:
"""

    query_disambiguation = """
Your task is to help the AI analyst tool determine the relevant columns given a user's query. You need to list all columns necessary to answer the query.

If the question is unclear and you are not sure which columns are meant, output 'none'. For example, if the user asks a question about voltage, but there are three types of voltages (1, 2, 3) and it is not clear which one is meant, output 'none'.

Examples:

Example 1:
Columns: ChargingCycleId, CarType, SoC_Start (pc), SoC_End (pc), Temperature (C), Current (A)
Data:
1, Taycan, 21, 24, 12, 160
2, Taycan, 26, 52, 22, 10
3, Taycan, 0, 8, -9, 232
Query: What is the highest temperature?
Answer: ['Temperature (C)']

Example 2:
Columns: ChargingCycleId, CarType, SoC_Start (pc), SoC_End (pc), Temperature (C), Current (A)
Data:
1, Taycan, 21, 24, 12, 160
2, Taycan, 26, 52, 22, 10
3, Taycan, 0, 8, -9, 232
Query: What is the highest SoC?
Answer: none

Example 3:
Columns: index, distance, mileage_start, mileage_end, speed_mean, speed_abs_diff_mean, speed_max, voltage_mean, voltage_first, voltage_last, payload_ts_first, payload_ts_last, soc_start, soc_end, soc_diff, current_abs_diff_mean, current_mean, current_abs_mean, car_type, records_length, power_kw_abs_mean, power_kw_mean, power_kw_median, power_kw_max, power_kw_min, energy_neg, energy_pos, temperature_ambient_first, temperature_ambient_last, temperature_ambient_min, temperature_ambient_max, temperature_ambient_mean, temperature_start, temperature_end, acceleration_neg_mean, acceleration_pos_mean, soc_on_km, data_source, sampling_rate, country_code_alpha2, timezone, payload_ts_first_formatted, payload_ts_last_formatted, year_month, str_ts_first, str_ts_last, weekday, perc_missing_data, vin
Data:
0, 3.0, 23483.0, 23486.31047781128, 65.69223445442073, 14.567049652667947, 82.59740423578664, 809.7003595778169, 795.0043656227721, 719.3353641427431, 1674420711276, 1682247127450, 70.10529833324077, 56.10890525860751, 20.68731732271334, 16.063066679617013, -31.473484327601923, 23.757686644370697, Taycan33, 23, 15.08699561467924, -12.02608751671494, -16.706308668622643, 0.6563081459552897, -67.01769102402076, -12.178321633633074, 0.0010194595535583, 7.458932692957482, 9.101118877040903, 17.216099296549388, 7.643567408807364, 8.87619815576264, 9.03155771028374, 9.43906646374834, -0.1179969381026706, 0.1422525936555543, 0.3263216922582791, ZDS_ucvcol, 60, PE, Europe/Brussels, 2023-03-06 14:25:52, 2023-01-31 11:39:05, 202302, 2022-11-03 14:37:12, 2023-05-24 09:48:34, Saturday, 0.0, 2290
Query: What is the average ambient temperature in the data?
Answer: none

Example 4:
Columns: index, distance, mileage_start, mileage_end, speed_mean, speed_abs_diff_mean, speed_max, voltage_mean, voltage_first, voltage_last, payload_ts_first, payload_ts_last, soc_start, soc_end, soc_diff, current_abs_diff_mean, current_mean, current_abs_mean, car_type, records_length, power_kw_abs_mean, power_kw_mean, power_kw_median, power_kw_max, power_kw_min, energy_neg, energy_pos, temperature_ambient_first, temperature_ambient_last, temperature_ambient_min, temperature_ambient_max, temperature_ambient_mean, temperature_start, temperature_end, acceleration_neg_mean, acceleration_pos_mean, soc_on_km, data_source, sampling_rate, country_code_alpha2, timezone, payload_ts_first_formatted, payload_ts_last_formatted, year_month, str_ts_first, str_ts_last, weekday, perc_missing_data, vin
Data:
0, 3.0, 23483.0, 23486.31047781128, 65.69223445442073, 14.567049652667947, 82.59740423578664, 809.7003595778169, 795.0043656227721, 719.3353641427431, 1674420711276, 1682247127450, 70.10529833324077, 56.10890525860751, 20.68731732271334, 16.063066679617013, -31.473484327601923, 23.757686644370697, Taycan33, 23, 15.08699561467924, -12.02608751671494, -16.706308668622643, 0.6563081459552897, -67.01769102402076, -12.178321633633074, 0.0010194595535583, 7.458932692957482, 9.101118877040903, 17.216099296549388, 7.643567408807364, 8.87619815576264, 9.03155771028374, 9.43906646374834, -0.1179969381026706, 0.1422525936555543, 0.3263216922582791, ZDS_ucvcol, 60, PE, Europe/Brussels, 2023-03-06 14:25:52, 2023-01-31 11:39:05, 202302, 2022-11-03 14:37:12, 2023-05-24 09:48:34, Saturday, 0.0, 2290
Query: Make a subtable of index, weekday, vin, mean start and last voltages and all SoCs. Include also all temperatures.
Answer: ['index', 'weekday', 'vin', 'voltage_mean', 'voltage_first', 'voltage_last', 'soc_start', 'soc_end', 'soc_diff', 'temperature_ambient_first', 'temperature_ambient_last', 'temperature_ambient_min', 'temperature_ambient_max', 'temperature_ambient_mean', 'temperature_start', 'temperature_end']

Guidelines:
1. If there are multiple columns with very similar names (e.g., last_temperature, start_temperature), and it is not clear which columns the user wants, output 'none'.
2. Some queries may require additional reasoning. Think about each query and check if the hidden meaning clearly determines specific columns.
3. Be very precise. Never make up your own column names.
4. If your output is a list, it must be loadable by json.loads; hence do not output anything else than the list.
5. List only the columns that are necessary to answer the query. Listing unnecessary columns is considered an error.
6. Beware of all the nuances. Some columns may have semantically very similar names. Make sure to list only the correct ones.
7. If you are not sure whether the user query is clear, output 'none'. But try to assume that the user query is clear.
8. Beware of what the user wants. Column names can include aggregation functions (e.g., 'mean temperature'). Segment what the user wants and the column names.
9. Determiners matter: 'all <column_name>' means all columns with the name <column_name>, 'the <column_name>' means only one column with the name <column_name>.
10. Singular/plural matters: 'temperature' means one single column, 'temperatures' means multiple columns.
11. Words like 'compute', 'calculate' together with an aggregation function (e.g., 'compute mean temperature') mean the user wants the aggregation function of the column.
12. Assume that unknown values (secret code, etc.) are not present in the column names and are actually known or given.
13. Often, the user query determines the column names implicitly. Do not output 'none' when the query is clear but columns are described implicitly.
14. Always output just the list or 'none'. Nothing else is allowed.

Our table has these columns:
{column_description}
Your Task:
{df_head}
Query: '{input}'
Answer:
"""

    def get_df_heads_str(self, df: pd.DataFrame | list[pd.DataFrame], head_number: int) -> str:
        if isinstance(df, pd.DataFrame):
            res = f'df:\n{df.head(head_number)}'
        else:
            res = ''
            for i, df in enumerate(df):
                res += f'df_{i + 1}:\n{df.head(head_number)}'

    def format_generate_steps_no_plot_prompt(self, head_number, df, user_query, column_description, functions_description):
        return self.generate_steps_no_plot.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            functions_description=functions_description
        )
    
    def format_reformulate_plan_prompt(self, head_number, df, user_query, plan, column_description, functions_description):
        return self.generate_steps_no_plot.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            plan=plan, 
            functions_description=functions_description
        )

    def format_generate_steps_for_plot_save_prompt(self, head_number, df, user_query, save_plot_name, column_description, functions_description):
        return self.generate_steps_for_plot_save.format(
            input=user_query, 
            plotname=save_plot_name, 
            df_head=self.get_df_heads_str(df, head_number), 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_steps_for_plot_show_prompt(self, head_number, df, user_query, column_description, functions_description):
        return self.generate_steps_for_plot_show.format(
            input=user_query, 
            df_head=self.get_df_heads_str(df, head_number), 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_code_prompt(self, head_number, df, user_query, plan, column_description, functions_description):
        return self.generate_code.format(
            input=user_query, 
            df_head=self.get_df_heads_str(df, head_number), 
            plan=plan, 
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_generate_code_for_plot_save_prompt(self, head_number, df, user_query, plan, column_description, functions_description, save_plot_name=""):
        return self.generate_code_for_plot_save.format(
            input=user_query, 
            df_head=self.get_df_heads_str(df, head_number), 
            plan=plan, 
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )

    def format_fix_code_prompt(self, head_number, df, user_query, code_to_be_fixed, error_message, column_description, functions_description):
        return self.fix_code.format(
            code=code_to_be_fixed, 
            df_head=self.get_df_heads_str(df, head_number), 
            error=error_message, 
            input=user_query, 
            head_number=head_number, 
            column_description=column_description, 
            functions_description=functions_description
        )
    
    def format_query_rewrite(self, head_number, df, user_query, column_description, functions_description):
        return self.query_rewrite.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            functions_description=functions_description
        )
    
    def format_is_query_clear(self, head_number, df, user_query, column_description, functions_description):
        return self.is_query_clear.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            # functions_description=functions_description
        )
    
    def format_generate_code_multiple_dfs_prompt(self, head_number, dfs, user_query, plan, column_descriptions, functions_description):
        data = "\n"
        for i in range(len(dfs)):
            data += f"DataFrame df_{i+1}:\n{dfs[i].head(head_number)}\n{column_descriptions[i]}\n"

        return self.generate_code_multiple_dfs.format(
            input=user_query, 
            plan=plan,
            functions_description=functions_description,
            data=data,
            df_examples=", ".join([f"df_{i+1}" for i in range(len(dfs))])
        )

    def format_query_disambiguation(self, head_number, df, user_query, column_description, functions_description):
        return self.query_disambiguation.format(
            df_head=self.get_df_heads_str(df, head_number), 
            input=user_query, 
            column_description=column_description, 
            # functions_description=functions_description
        )
