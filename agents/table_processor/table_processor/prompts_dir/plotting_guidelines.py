plotting_best_practices = """
Ensure the plot has clear labels for the x and y axes.
Use a dynamic title that reflects specific features of the data.
If applicable, use a grid to enhance plot readability.
Choose a color scheme that is accessible to all audiences, including those with color vision deficiencies.
When plotting time-series data, format the time axis for clarity and context.
Optimize tick mark labels and their frequency to improve readability and prevent overlap.
Implement legends effectively to ensure they are clear and do not obstruct the data visualization.
Use annotations to highlight specific points or trends in the data, providing context where necessary.
Employ consistent and appropriate font sizes for all text elements to enhance readability.
Consider using transparency in dense plot areas to help distinguish overlapping data points.
Utilize subplots to compare several related charts in one figure, keeping a coherent layout and scale.
Incorporate error bars in plots to represent the variability or uncertainty in the data.
Choose plot types that best represent the underlying patterns and relationships in the data.
When plotting data involving days of the week, ensure the x-axis orders weekdays from Monday to Sunday for intuitive understanding.
"""

seaborn_implementation_old = """
Ensure the plot has clear labels for the x and y axes using Seaborn's label functions.
Use Seaborn to set a dynamic title that reflects specific features of the data, ensuring the title is slightly larger than the axes labels for improved visibility.
Utilize Seaborn's grid settings ('whitegrid') to enhance plot readability.
Choose a Seaborn theme that is accessible to all audiences, including those with color vision deficiencies.
Format time axes clearly when plotting time-series data with Seaborn.
Optimize tick mark labels and their frequency using Seaborn's control over Matplotlib.
Use Seaborn's legend functionality to ensure clear and non-obstructive data visualization.
Highlight specific points or trends with annotations using Seaborn's capabilities.
Maintain consistent and appropriate font sizes for all text elements in the plot, with the title being slightly larger.
Apply transparency in dense plot areas with Seaborn to distinguish overlapping data points.
Use Seaborn's FacetGcarid for subplots to compare several related charts in one coherent layout.
Incorporate error bars in plots using Seaborn's error bar capabilities to represent variability.
Select plot types from Seaborn that best represent the data's underlying patterns and relationships.
Initialize the plot canvas only once before adding data to prevent multiple plots being generated unintentionally.
Save the plot immediately after creation to ensure all changes are captured in a single output file.

Examples of Seaborn visualizations:
1. **Bar Chart**: Use `sns.barplot` to compare categorical data. Customize with `palette` for color and `errorbar` to show variability.
2. **Line Chart**: Utilize `sns.lineplot` for time-series data, which automatically handles trend visualization with confidence intervals.
3. **Pie Chart**: Although Seaborn does not supportd pie charts, apply Seaborn styles using `sns.set()` before creating pie charts with Matplotlib to keep the style consistent.
4. **Scatter Plot**: Implement `sns.scatterplot` for bivariate distributions, adding `hue`, `style`, and `size` parameters to differentiate subgroups.
5. **Histogram**: Use `sns.histplot`, which combines histogram and kernel density estimate plots, offering deeper insights into distributions.

For each plot, ensure to:
- Use clear labels for axes and an informative title.
- Apply a grid for better readability.
- Use annotations to highlight important data points.
- Maintain aesthetic consistency with Seaborn's theming capabilities.
"""

seaborn_implementation = """
Ensure the plot has clear labels for the x and y axes by utilizing Seaborn's set method to integrate labels directly into the plot functions for maintaining style consistency.
Set a dynamic title that reflects specific features of the data using the title parameter within Seaborn's plotting functions, ensuring the title is prominently visible and stylistically integrated.
Employ Seaborn's 'set_theme()' at the beginning of your plotting session to ensure consistent application of styles throughout your plots, including the grid setting which can be set to 'whitegrid' for better readability.
Apply a Seaborn color palette that is accessible to all audiences, including those with color vision deficiencies, by using 'color_palette()' or 'set_palette()' methods right before plotting data.
Optimize tick mark labels and their frequency by adjusting Seaborn's scale and label settings within the plotting functions to ensure labels are clear and non-overlapping.
Use Seaborn's built-in functions for legends and annotations to ensure they are seamlessly integrated into the plots without obstructing visual data interpretation.
Maintain uniform font sizes and styles across all text elements in the plot by managing the font scale using Seaborn's 'set_context()' function, tailored for various types of presentations (paper, talk, poster).
Incorporate transparency in plots where data points overlap by utilizing the 'alpha' parameter within Seaborn's plotting functions to differentiate overlapping data points effectively.
Utilize Seaborn's 'FacetGrid' or 'PairGrid' for creating subplots that compare multiple variables simultaneously, ensuring a coherent and consistent layout across multiple plots.
Apply error bars within your plots using Seaborn's built-in support in functions like 'barplot', 'pointplot', etc., to visually represent data variability or uncertainty.
Choose appropriate plot types from Seaborn's extensive library that best visualize the underlying patterns and relationships in the data, ensuring each plot type is used in its context.
Initiate your plot setups by configuring Seaborn settings at the start of your script to apply a uniform style across all subsequent plots.
For saving plots:
- Always save the figure using `plt.savefig', regardless of whether the plot was created using Seaborn or Matplotlib.
- Avoid using `.get_figure().savefig()` even for Seaborn plots to ensure consistent behavior.
- After saving the plot, clear the current figure using `plt.clf()` to prevent overlap with subsequent plots.

Examples of Seaborn visualizations:
1. **Bar Chart**: 
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   bar_plot = sns.barplot(x=df.index, y='soc_end', data=df, ci=None, alpha=0.8)
   bar_plot.set_xlabel('Index')
   bar_plot.set_ylabel('State of Charge End (%)')
   bar_plot.set_title('Bar Chart of State of Charge End (soc_end)')

   # Save the plot
   plt.savefig('temp/0_SEG_plot_3ddcb75b928c153e94419c0cb9adfe21339.png')
   plt.clf()
   
Examples of Seaborn visualizations:
1. **Bar Chart**: Configure Seaborn at the start with 'sns.set_theme()', then create a bar chart using 'sns.barplot()', adjusting 'palette' and 'errorbar' directly.
2. **Line Chart**: After setting the theme, use 'sns.lineplot()' to create a line chart, perfect for time-series data, with confidence intervals automatically managed.
3. **Scatter Plot**: Begin by configuring Seaborn settings, then use 'sns.scatterplot()' to examine bivariate distributions, customizing 'hue', 'style', and 'size' to enhance subgroup distinctions.
4. **Histogram**: Start with 'sns.set()', then use 'sns.histplot()' to analyze distributions, combining histograms with kernel density estimates for deeper insights.


Remember to:
- Integrate labels and titles directly through Seaborn to ensure consistency.
- Manage grid lines and color schemes from the onset of plot creation using Seaborn's comprehensive settings.
- Embed annotations and legends using Seaborn's native functionalities to maintain the aesthetic integrity of plots.
"""



extra_prompts = """
Set a figure size that makes the plot easily readable, using Matplotlib's 'figsize' attribute in conjunction with Seaborn.
"""
