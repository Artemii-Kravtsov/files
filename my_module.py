
#%%
def comb_no_rep(n, k):
    if 'math' not in dir():
        from math import factorial
    return int(factorial(n) / (factorial(n-k) * factorial(k)))
def perm_no_rep(n, k):
    if 'math' not in dir():
        from math import factorial
    return int(factorial(n) / factorial(n-k))
def perm_with_rep(n, k):
    return n**k
def comb_with_rep(n, k):
    if 'math' not in dir():
        from math import factorial
    return int(factorial(n+k-1) / (factorial(n-1) * factorial(k)))
def factor(n):
    if 'math' not in dir():
        from math import factorial
    return factorial(n)
#%%
def grapefruit():
    """
    Возвращает для экспериментов dataframe со случайными 
    значениями в столбцах разных типов, объём (1000, 9). 
    """
    if not any(('pandas' in dir(), 'pd' in dir())):
        import pandas as pd
    if not any(('numpy' in dir(), 'np' in dir())):
        import numpy as np
    
    df = pd.DataFrame({
    'ints': np.random.randint(1000, 9999, 1000),
    'small_ints': np.random.randint(-1, 3, 1000),
    'floats': np.random.uniform(1, 1000, 1000) })

    alphabet = np.array(['y', 's', 'd', 'u', 'e', 'o', 'a'])
    df['strings'] = [''.join(np.random.choice(alphabet, size = 3)) for i in range(1000)]
    df['lists'] = [ [np.random.randint(1, 9), ''.join(np.random.choice(alphabet, size = 2))] for q in range(1000) ]
    df['booleans'] = df['small_ints'].astype(bool)
    df['datetime'] = [pd.to_datetime('01-03-2020', dayfirst = True) + pd.to_timedelta(np.random.randint(0, 120), unit='d') for i in range(1000)]
    df['categories'] = pd.cut(df['ints'], 8, labels = ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'))
    df['nans'] = df[ df['ints'] > np.random.randint(3000, 8000) ]['small_ints']
    return df
#%%
def convert_excel_datetime(df, column):
    """
    Принимает df и название столбца с датой. Преобразует дату,
    записанную ранее в файле excel в формате числа в 
    формат datetime. 
    """
    if 'pandas' not in dir() or 'pd' not in dir():
        import pandas as pd

    return pd.to_datetime('1900-01-01') + pd.to_timedelta(df[column] - 2, unit='d')

#%%
def wipe_globals(*but_keep_those):
    """
    Функция выполняет магические команды %reset_selective и %xdel, 
    очищая память ото всех глобальных переменных, кроме тех, которые
    отсылают к основным рабочим библиотекам и тех, которые присутствуют 
    в памяти сразу после запуска. Функция примет любое количество 
    строк с именами - объекты под этими именами будут защищены от удаления.
    Себя также не удаляет.
    """
    if not 're' in dir():
        import re
    if not 'get_ipython' in dir():
        from IPython import get_ipython
    from gc import collect
    
    keep_those_too = '_(i|o|d|n)(1|h|i|m)?(i|s)?$|re$|tf$|quit$|exit$|jtplot$|wipe_globals$|sns$|sklearn$|scipy.fftpack$|req$|robjects$|nltk$|get_ipython$|__.*$|NamespaceMagics$|_Jupyter$|json$|getsizeof$|var_dic_list$|_$|__$|np$|mm$|pd$|plt$|In$|Out$).*'
    regex_filter = ('^(?!' 
                    + ''.join([f'{x}$|' for x in but_keep_those]) 
                    + ''.join([f'{x}$|' for x in list(locals())]) 
                    + keep_those_too)
    
    already_removed = []
    for x in globals().copy():
        if x in already_removed:
            continue
        if re.match(regex_filter, x):
            try:
                get_ipython().run_line_magic('xdel', x)
                already_removed.append(x)
            except:
                pass
    
    print('Global namespace has been cleared')
    
    get_ipython().run_line_magic('reset_selective', regex_filter)
    collect()
    return
#%%
def downcast_convertion(df):
    """
    К числовым столбцам применяет to_numeric(downcast = ...), 
    Строковые столбцы переводит в формат category, если в них 
    не более тридцати процентов значений уникальные.
    Возвращает обновлённый dataframe.
    """
    if not any(('pandas' in dir(), 'pd' in dir())):
        import pandas as pd
    
    if len(df.select_dtypes(include = ['float64']).columns) > 0:
        for column_name in df.select_dtypes(include = ['float64']).columns:
            df[column_name] = pd.to_numeric(df[column_name], errors = 'raise', downcast = 'float')

    if len(df.select_dtypes(include = ['int64']).columns) > 0:    
        for column_name in df.select_dtypes(include = ['int64']).columns:
            if df[column_name].min() >= 0: 
                df[column_name] = pd.to_numeric(df[column_name], errors = 'raise', downcast = 'unsigned')
            else: 
                df[column_name] = pd.to_numeric(df[column_name], errors = 'raise', downcast = 'signed')     

    if len(df.select_dtypes(include = ['object']).columns) > 0:    
        for column_name in df.select_dtypes(include = ['object']).columns:
            if len(df[column_name].unique()) / 0.3 < len(df[column_name]):
                df[column_name] = df[column_name].astype('category')
                
    return df
#%%
def gaps_duplicated(df):
    """
    Принимает dataframe. Возвращает два dataframe'f, в первом котором обозначены пары столбцов,
    имеющиех более 40%, но менее 90% совпадающих строк с пропусками, во втором - более 90%.
    Сравнивает объединение стоолбцов с пропусками с пересечением столбцов с пропусками.
    """
    if not any(('pandas' in dir(), 'pd' in dir())):
        import pandas as pd
    if not 'display' in dir():
        from IPython.display import display
   
    matches_full = pd.DataFrame(index = df.isna().sum()[df.isna().sum() > 0].index,
                            columns = df.isna().sum()[df.isna().sum() > 0].index)
    
    matches_part = pd.DataFrame(index = df.isna().sum()[df.isna().sum() > 0].index,
                            columns = df.isna().sum()[df.isna().sum() > 0].index)
    
    for i in matches_full.columns:
        for j in matches_full.index:

            if i == j:
                continue

            match_rate = df[ df[j].isna() & df[i].isna() ].shape[0] / df[ df[j].isna() | df[i].isna() ].shape[0]
            if match_rate > 0.9:
                matches_full.loc[j, i] = '{:.2%} match'.format(match_rate)
            elif match_rate > 0.4:
                matches_part.loc[j, i] = '{:.2%} match'.format(match_rate)
                
    matches_part = matches_part.dropna(how = 'all', axis = 0).dropna(how = 'all', axis = 1).fillna('-')
    matches_full = matches_full.dropna(how = 'all', axis = 0).dropna(how = 'all', axis = 1).fillna('-')

    display(matches_part)
    display(matches_full)
#%%
def gaps_histograms(df, main, *columns, datetime = '', rotation = 'horizontal'):
    """
    Первым аргументом принимает датафрейм. Вторым аргументом принимает название столбца, содержащего пропуски.
    Затем список колонок, для которых будет построен рисунок. В параметре 'datetime =' можно
    через пробел передать названия datetime столбцов c пропусками (лучше заранее перевести вручную, имея 
    в виду возможные ошибки "автоматического" перевода без указания параметра 'format ='). Функция округлит
    значение datetime до даты. Рисунок будет состоять из двух наложенных друг на друга гистограмм. Одна гистограмма (голубого цвета) показывает распределение 
    значений строчек, у которых нет пропусков в столбце main. Вторая (красного цвета) - тех, где есть пропуски.
    Для того чтобы привести гистограммы к одному масштабу (т.к. строк с пропусками заведомо меньше),
    гистограммы отражают процентное распределение значений (density) - до от 0% до 70%. Отображать выше 70% - 
    делало бы гистограмму менее наглядной, при том что семидесяти процентов в целом достаточно, чтобы понять,
    обнаружилась ли какая-то тенденция. Для строковых столбцов функция строит гистограмму для двадцати самых частых 
    уникальных значений, для числовых и datetime - гистограмму для интервалов (способы выделить корзины и их число
    отображаются в заголовке гистораммы). Если какое-то значение в одном срезе встречается на 8%+ чаще, чем 
    в другом, оно текстом появляется в соответствующем месте на гистограмме. Если текст не видно, его можно
    наклонить, передав параметр rotation= .
    """
    if 'pd' not in dir():
        import pandas as pd
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'display' not in dir():
        from IPython.display import display
        
    df = df.copy()

    import matplotlib.patheffects as path_effects
    style = dict(size = 13, color = '#929591', rotation = rotation, 
                 path_effects = [path_effects.SimpleLineShadow(linewidth = 2, foreground = 'black'), 
                                 path_effects.Normal()])
    
    leng = len(columns) + len(datetime.split())
    
    nrows = [leng // 3 + 1 if leng % 3 != 0 else leng // 3][0]
    # here list comprehension calculates how many rows do we need (3 axes per row) 
    fig_height = 4 * nrows
    fig = plt.figure(figsize = (18, fig_height))
    grid = fig.add_gridspec(nrows, 3, wspace = 0.2, hspace = 0.3)
    # gridspec is a figure pattern that will help me to locate axes properly
    axes_made = 0
    for row in range(nrows):
        for col in range(3):
            if axes_made < leng:
                ax = fig.add_subplot(grid[row, col])
                # i create axes on gridspec coordinates until i've created enough
                ax.tick_params(labelbottom = False, labelleft = False, grid_alpha = 0.5)
                ax.set(ylim = (0, 0.7), ylabel = '0.0  <----  density  ---->  0.7')
                axes_made += 1
    ax = fig.axes
    # i place all the axes at the variable 'ax'. now i can acess axes via iterating over 
    # 'leng' variable, as long as their ('ax' and 'leng') lengths completely match 
    
    columns = list(columns)
    # the function provides user arguments in tuple format. before making changes
    # on it, we have to make a list out of it
    
    datetime_columns = []        
    if datetime != '':
        datetime = datetime.split()
        for dt_col in datetime:
        # if we don't convert a datetime column with NaN's from object-type to datetime,
        # we can't cut it via cut( ) or qcut( ). dropna( ) is not an option - new NaN's 
        # will instantly replace the dropped ones. i see two solutions to the problem: to fill NaN's
        # with an outlying, easily distinguished value (and to make a convertion now) or to
        # mark those columns out and to convert them as soon as we take their values out
        # of dataframe. as for me, the second solution is more consistent with plotting goals.
            df[dt_col + '_'] = df[dt_col].astype('object')
            datetime_columns.append(dt_col + '_')
            columns.append(dt_col + '_')
    
    category_columns = df.select_dtypes(include=['category']).columns
    for cat_col in category_columns:
        if cat_col in columns:
            df[cat_col + '_'] = df[cat_col].astype('object')
            # i am creating new object-type columns out of category ones,  
            # otherwise i won't be able to plot them the way i want. and i remove
            # category columns from the list of columns to be plotted
            columns.remove(cat_col)
            columns.append(cat_col + '_')
            
    def customizing_numeric_intervals(interval):
        interval = str(interval)
        left, right = interval.split()
        left_integer, left_decimals = left.split('.')
        left_integer = left_integer
        left_decimals = left_decimals[0]
        right_integer, right_decimals = right.split('.')
        right_decimals = right_decimals[0]
        return ''.join([left_integer, '.', left_decimals, '-', right_integer, '.', right_decimals + ']'])
    def customizing_datetime_intervals(interval):
        interval = str(interval)
        left, right = interval.split(', ')
        if len(left.split(' ')) == 2:
            left_date, left_time = left.split(' ')
            left_date = left_date[1:]
        else:
            left_date = left
        if len(right.split(' ')) == 2:
            right_date, right_time = right.split(' ')
        else:
            right_date = right
        return ''.join([left_date, ' / ', right_date])   
    
    for i in range(leng):
        
        if columns[i] in datetime_columns:
            is_datetime = True
        else:
            is_datetime = False
    
        if (df[columns[i]].dtype != 'O' or df[columns[i]].dtype != 'object') or is_datetime == True:
            
            if is_datetime == False:
                if (pd.cut(df[columns[i]], 20, duplicates = 'drop')
                    .value_counts(normalize = True)
                    .iloc[:4].sum() > 0.8):
                    # if after cut( ) operation first four elements accumulate more
                    # than 80% of values, a histogram won't be much illustrative. 
                    # in this case dividing by quantiles is more preferable.                
                    df[f'{columns[i]}_'] = pd.qcut(df[columns[i]], 10, duplicates = 'drop')
                    title = 'qcut'
                    # if the division by 20 quantiles is steady, an average occurence rate
                    # of a category will be 5%. a histogram won't be  
                    # very much illustrative as well. so let's divide by 10 quantiles.   
                else:
                    df[f'{columns[i]}_'] = pd.cut(df[columns[i]], 20, duplicates = 'drop')
                    title = 'cut'
            elif is_datetime == True:
                title = 'qcut'
                df[f'{columns[i]}_'] = pd.qcut(pd.to_datetime(df[ df[columns[i]].notna() ][columns[i]]), 
                                               10, duplicates = 'drop')
                        
            missing = df[ df[main].isna()][f'{columns[i]}_'].cat.codes
            normal = df[ df[main].notna()][f'{columns[i]}_'].cat.codes           
            # after cut( ) and qcut( ) we have to deal with a category column. 
            # it is impossible to plot intervals, though we can plot integers that 
            # substitute them in memory. 
            
            # both (q)cut( ) and hist( ) ignore NaN's in a slice and do their job as they
            # sould, but as far as i want to set labels that indicate how many rows are covered by
            # a histogram, i have to substract missing values from slice manually.
            # i need to mark them as a category, other than 'out_of_any_category_but_still_not_a_gap', 
            # as they are in default in category-type columns. or i can request the count - that sounds easier.
            # when NaN's are converted to category-type, they are stored under number -1. 
            try: 
                gaps_in_missing = missing.value_counts().loc[-1]
            except KeyError:
                gaps_in_missing = 0
            try:
                gaps_in_normal = normal.value_counts().loc[-1]
                bins = df[f'{columns[i]}_'].unique().shape[0] - 1
                # if cut( ) or qcut( ) have had a parameter bins = 15 and slice has 
                # contained NaNs, resulting column now has 16 unique values, one of which 
                # won't be plotted. if so, the number of bins that i should pass is 15, not 16. 
            except KeyError:
                gaps_in_normal = 0
                bins = df[f'{columns[i]}_'].unique().shape[0]
            
            title = title + f'({bins}): {columns[i]}'
            label_for_miss = f'{missing.shape[0] - gaps_in_missing} rows'
            label_for_norm = f'{normal.shape[0] - gaps_in_normal} rows'
            
            ax[i].hist(normal, histtype = 'bar', bins = bins, density = True, 
                    color = '#01ecb1', alpha = 0.8, label = label_for_norm)
            ax[i].hist(missing, histtype = 'bar', bins = bins, density = True, 
                    color = '#ec2d01', alpha = 0.7, label = label_for_miss)
            ax[i].legend()
            ax[i].set(title = title)
            
            misses_for_inscr = df[ df[main].isna() ][f'{columns[i]}_'].value_counts(normalize = True)
                # this block of code is dedicated to making inscriptions on places where two histograms dispart
            normals_for_inscr = df[ df[main].notna() ][f'{columns[i]}_'].value_counts(normalize = True)
            for_inscr = pd.concat([normals_for_inscr, misses_for_inscr], axis = 1)
                # 'for_inscr' is a dataframe that keeps the occurance rate of every 
                # value in two different slices. slices are identical to those that provide 
                # a base for histograms: one slice - rows with gaps, another - rows without gaps.
            for_inscr['difference'] = abs(for_inscr.iloc[:, 0].sub(for_inscr.iloc[:, 1]))
                # adding modular subtraction of occurence rates in two slices.
                # the resulting number doesn't refer to anything real, but it reflects
                # the very difference in bars 
            for_inscr = (for_inscr[ for_inscr['difference'] > 0.08 ]
                         .sort_values(by = 'difference', ascending = False).head(3))
                # here i'm releasing all values except those three of them that differ the most. 
                # since there is no need in making an indcation when the difference 
                # is still too low, i've decided to set 10% (0.1) threshold
            for_inscr['maximal'] = for_inscr.iloc[:, :2].max(axis = 1)
                # maximal occurance rate will be a coordinate on Y-axis where to place. 
                # an inscription. still it's not a density, but something proportional
            for_inscr['maximal'] = for_inscr['maximal'].where(for_inscr['maximal'] < 0.65, 0.65)
                # y-axis is constrained by 0.7, while occurence rate is not.
                # suppose, we don't want to place an inscription above the drawin
            for_inscr = for_inscr.drop(f'{columns[i]}_', axis = 1)
            for_inscr = for_inscr.reset_index()
                # after .reset_index( ) we got a new column accessible via .iloc[:, 0]
            for_inscr['codes'] = for_inscr.iloc[:, 0].cat.codes
            if is_datetime:
                for_inscr['inscription'] = for_inscr.iloc[:, 0].apply(customizing_datetime_intervals)
            else:
                for_inscr['inscription'] = for_inscr.iloc[:, 0].apply(customizing_numeric_intervals)
                

            if for_inscr.shape[0] > 0:
                for y in range(for_inscr.shape[0]):
                    ax[i].text(for_inscr.loc[y, 'codes'], 
                               for_inscr.loc[y, 'maximal'],
                               "{}: {:.0f}prc".format(for_inscr.loc[y, 'inscription'], (for_inscr.loc[y, 'difference'] * 100)),
                               **style)
            
        else:
            top = df.groupby(columns[i])[columns[i]].count().sort_values(ascending = False).index[:20]
            missing = df[ df[main].isna() & df[columns[i]].notna() ].query(f'{columns[i]} in @top')[columns[i]]
            normal = df[ df[main].notna() & df[columns[i]].notna() ].query(f'{columns[i]} in @top')[columns[i]]
            if len(top) < 20:
                # we've finished using 'top' variable to make slices.
                # now we can use it to store the number of bins
                top = len(top)
            else:
                top = 20
            label_for_miss = f'{missing.shape[0]} rows'
            label_for_norm = f'{normal.shape[0]} rows'
            title = f'top({top}): {columns[i]}'
            ax[i].hist(normal, histtype = 'bar', bins = top, density = True, 
                    color = '#01ecb1', alpha = 0.8, label = label_for_norm)
            ax[i].hist(missing, histtype = 'bar', bins = top, density = True, 
                    color = '#ec2d01', alpha = 0.7, label = label_for_miss)
            ax[i].legend()
            ax[i].set(title = title)
            
        
            misses_for_inscr = df[ df[main].isna() ][f'{columns[i]}'].value_counts(normalize = True)
            normals_for_inscr = df[ df[main].notna() ][f'{columns[i]}'].value_counts(normalize = True)
            for_inscr = pd.concat([normals_for_inscr, misses_for_inscr], axis = 1).fillna(0)
                # after concateration we will compare values that are present in both
                # slices, others will return NaN. i replace them with 0 to calculate the difference
            for_inscr['difference'] = abs(for_inscr.iloc[:, 0].sub(for_inscr.iloc[:, 1]))
            for_inscr = (for_inscr[ for_inscr['difference'] > 0.08 ]
                         .sort_values(by = 'difference', ascending = False).head(3))
            for_inscr['maximal'] = for_inscr.iloc[:, :2].max(axis = 1) 
            for_inscr['maximal'] = for_inscr['maximal'].where(for_inscr['maximal'] < 0.60, 0.60) 
            for_inscr = for_inscr.reset_index()

            if for_inscr.shape[0] > 0:
                for y in range(for_inscr.shape[0]):
                    ax[i].text(for_inscr.loc[y, 'index'], 
                               for_inscr.loc[y, 'maximal'], 
                               "{}: {:.0f}prc".format(for_inscr.loc[y, 'index'], (for_inscr.loc[y, 'difference'] * 100)),
                               **style)
    display(fig)
    plt.close(fig)
            
    for x in df.columns:
        symbols = list(x)
        if symbols[-1] == '_':
            df = df.drop(x, axis = 1)
            # i'm dropping auxiliary columns with suffix '_' at the end of their names
#%%    
def gaps_investigation(df):
    """
    Принимает dataframe. Возвращает сводку (в df) по всем столбцам, содержащим
    пропуски со следующей информацией: кол-во пропусков, кол-во пропусков 
    в процентах от общего числа значений в столбце, минимальное и 
    максимальное значение в столбце, кол-во уникальных значений (если их 
    не более 3% от общего числа), два столбца с наибольшей корреляцией
    и доля пропусков в коррелирующих столбцах в тех же строчках. 
    Исключает строковые стобцы из рассмотрения. Корреляция считается
    по методам Спирмана и Пирсона, но в результирующей таблице не уточняется,
    по какому методу получена данная корреляция.
    """
    if 'pd' not in dir():
        import pandas as pd
    if not 'display' in dir():
        from IPython.display import display
    
    indices = list(df.isna().sum()[ df.isna().sum() > 0 ].index)
    # names of columns which contain gaps are going to be set as indices
    for_check_up = list(df.select_dtypes(include=['object']))
    # you can never find a correlation to a string column or its' min( )/max( ). 
    # though we don't have those, i felt like emphasising it for future cases
    object_columns = []
    for x in indices.copy():
        if x in for_check_up:
            object_columns.append(x)
            indices.remove(x)
    if len(object_columns) > 0:
        print("I excluded following columns for being dtype='object':", object_columns)
    
    gaps = pd.DataFrame(index = indices, 
                        columns = ['total_gaps', 'pct_of_gaps', 'min_max', 'uniques',  
                                   'corr_max', 'gaps_resp_ly', 'runner_up', 'gaps_in_ru'])
    total_rows = df.shape[0]
    # we'll need it later

    for x in gaps.index:
        total_gaps = df[x].isna().sum()
        # we'll need it later #2

        gaps.loc[x, 'total_gaps'] = total_gaps
        gaps.loc[x, 'pct_of_gaps'] = '{:.0%}'.format(total_gaps / total_rows)

        category_columns = list(df.select_dtypes(include=['category']).columns)
         # you have to treat category columns differently to be able to apply
         # corr() to them. so i request a list of category columns
        all_other_cols = list(df.select_dtypes(exclude=['object']))
        all_other_cols.remove(x)
         # no need to compare a column to itself

        if x in category_columns:
            gaps.loc[x, 'uniques'] = df[x].cat.categories.shape[0]
        else:
            min_max = str(round(df[x].min(), 1)) + ' / ' + str(round(df[x].max(), 1))
            gaps.loc[x, 'min_max'] = min_max
            uniques = df[x].unique().shape[0]
            if uniques / total_rows < 0.03:
            # a small amount of unique values may lead us to describe the variable as categorical
                gaps.loc[x, 'uniques'] = uniques

        corrs_dict = {}
        # i've created a dictionary. Key is a correlation rate between 
        # constant 'x' column and a 'column', value is 'column''s name
        for column in all_other_cols:
            if column in category_columns and x in category_columns:                
                try:
                    corr_rate = int(df[x].cat.codes.corr(df[column].cat.codes) * 100)
                    corr_rate_sp = int(df[x].cat.codes.corr(df[column].cat.codes, method = 'spearman') * 100)
                    # i make integers [-100, 100] out of floats [-1.0, 1.0] to avoid obstacles,
                    # typical for calculations with floats. there is no way i could possibly track
                    # NaN values as soon as they are returned. either way I receive a float. since NaN's
                    # aren't equal to anything, i can't even bring a conditional expression to track them
                except ValueError: 
                    corr_rate = 0
                    corr_rate_sp = 0
                    # ValueError means that one of columns contains just one constant value that
                    # gives a standard deviation (std) of 0. instead of dividing by 0,
                    # corr( ) function returns NaN. once trying to do int(NaN) we receive ValueError 
                    # for us it is equal to having 0.0 corr rate.                   
                corrs_dict[corr_rate] = corrs_dict.get(corr_rate, '') + ' ' + column
                corrs_dict[corr_rate_sp] = corrs_dict.get(corr_rate_sp, '') + ' ' + column
            elif column in category_columns and x not in category_columns:
                try:
                    corr_rate = int(df[x].corr(df[column].cat.codes) * 100)
                    corr_rate_sp = int(df[x].corr(df[column].cat.codes, method = 'spearman') * 100)
                except ValueError:
                    corr_rate = 0
                    corr_rate_sp = 0
                except TypeError:
                    corr_rate = int(df[x].dt.date.astype('category').cat.codes.corr(df[column].cat.codes) * 100)
                    corr_rate_sp = int(df[x].dt.date.astype('category')
                                       .cat.codes.corr(df[column].cat.codes, method = 'spearman') * 100)
                    # TypeError means the columns' type is datetime. it's a typical situation
                corrs_dict[corr_rate] = corrs_dict.get(corr_rate, '') + ' ' + column
                corrs_dict[corr_rate_sp] = corrs_dict.get(corr_rate_sp, '') + ' ' + column
            elif column not in category_columns and x in category_columns:
                try:
                    corr_rate = int(df[x].cat.codes.corr(df[column]) * 100)
                    corr_rate_sp = int(df[x].cat.codes.corr(df[column], method = 'spearman') * 100)
                except ValueError:
                    corr_rate = 0
                    corr_rate_sp = 0
                except TypeError:
                    corr_rate = int(df[x].cat.codes.corr(df[column].dt.date.astype('category').cat.codes) * 100)
                    corr_rate_sp = int(df[x].cat.codes.corr(df[column].dt.date
                                                            .astype('category').cat.codes, method = 'spearman') * 100)
                corrs_dict[corr_rate] = corrs_dict.get(corr_rate, '') + ' ' + column
                corrs_dict[corr_rate_sp] = corrs_dict.get(corr_rate_sp, '') + ' ' + column
            else:
                try:
                    corr_rate = int(df[x].corr(df[column]) * 100)
                    corr_rate_sp = int(df[x].corr(df[column], method = 'spearman') * 100)
                except ValueError:
                    corr_rate = 0
                    corr_rate_sp = 0
                except TypeError:
                    if df[x].dtype == '<M8[ns]' and df[column].dtype == '<M8[ns]':
                        corr_rate = int(df[x].dt.date.astype('category').cat.codes
                                        .corr(df[column].dt.date.astype('category').cat.codes) * 100)
                        corr_rate_sp = int(df[x].dt.date.astype('category').cat.codes
                                           .corr(df[column].dt.date.astype('category').cat.codes, 
                                                 method = 'spearman') * 100)
                    elif df[x].dtype == '<M8[ns]':
                        corr_rate = int(df[x].dt.date.astype('category').cat.codes.corr(df[column]) * 100)
                        corr_rate_sp = int(df[x].dt.date.astype('category').cat.codes
                                           .corr(df[column], method = 'spearman') * 100)
                    elif df[column].dtype == '<M8[ns]':
                        corr_rate = int(df[x].corr(df[column].dt.date.astype('category').cat.codes) * 100)
                        corr_rate_sp = int(df[x].corr(df[column].dt.date.astype('category').cat.codes, 
                                                      method = 'spearman') * 100)
                corrs_dict[corr_rate] = corrs_dict.get(corr_rate, '') + ' ' + column
                corrs_dict[corr_rate_sp] = corrs_dict.get(corr_rate_sp, '') + ' ' + column
   
        max_corr = []
        max_corr_columns = []
        corrs_list = list(corrs_dict.keys())
        while len(max_corr) < 2:
            max_corr_key = max(corrs_list, key = abs)
            # we've picked up two maximal correlations and saved them in max_corr
            # now let's put'em in 'gaps' table
            corrs_list.remove(max_corr_key)
            columns_max_corr = corrs_dict[max_corr_key].split()
            # as you remember, we added ' ' to every value in dictionary to separate them. now it is time to
            # deconstuct those strings. here we make a list of column(s) with identical corr_rate (i.e. key)
            for col in columns_max_corr:
                if col not in max_corr_columns:
                    max_corr.append((max_corr_key, col))
                    max_corr_columns.append(col)
    
        if max_corr[0][0] != 0:
            gaps.loc[x, 'corr_max'] = f'{max_corr[0][0]}: {max_corr[0][1]}' 
            gaps.loc[x, 'gaps_resp_ly'] = '{:.0%}'.format(df[ df[max_corr[0][1]].isna() & 
                                                              df[x].isna() ].shape[0] / total_gaps)
            # this will tell us how many rows in two correlated columns match in having gaps
        if max_corr[1][0] != 0:
            gaps.loc[x, 'runner_up'] = f'{max_corr[1][0]}: {max_corr[1][1]}' 
            gaps.loc[x, 'gaps_in_ru'] = '{:.0%}'.format(df[ df[max_corr[1][1]].isna() & 
                                                              df[x].isna() ].shape[0] / total_gaps)
    display(gaps.fillna('-'))
    if len(object_columns) > 0:
       print("Following columns were not included because of being 'object' dtype:")
       for x in object_columns:
           print(f"\t-'{x}' with {df[x].isna().sum()} gaps, which is {round(df[x].isna().sum() / df[x].shape[0], 1)}% of all values in this column")
#%%
def customizing_intervals(interval, dtype = '', decimals = 0, parenthesis = False):
    if (dtype != 'numeric') and (dtype != 'datetime'):
        print("Pass dtype= argument: 'numeric' or 'datetime'")
        return
    elif dtype == 'numeric':      
        interval = str(interval)
        left, right = interval.split()
        left_integer, left_decimals = left.split('.')
        right_integer, right_decimals = right.split('.')
        left_integer = left_integer[1:]   
        left_decimals = left_decimals[:decimals]
        right_decimals = right_decimals[:decimals]
        if parenthesis == False:
            return ''.join([left_integer, '.', left_decimals, ' - ', right_integer, '.', right_decimals])
        else:
            return ''.join(['(', left_integer, '.', left_decimals, ' - ', right_integer, '.', right_decimals, ']'])
    else:
        interval = str(interval)
        left, right = interval.split(', ')
        if len(left.split(' ')) == 2:
            left_date, left_time = left.split(' ')
            left_date = left_date[1:]
        else:
            left_date = left
        if len(right.split(' ')) == 2:
            right_date, right_time = right.split(' ')
        else:
            right_date = right
        return ''.join([left_date, ' / ', right_date])
#%%
def outliers_investigation(df, include = '', top_whisker = 1.5, bot_whisker = 1.5,
                           exclude = '', exclude_cat = '', zero = -909090, unknown = -777777):
    """
    Позиционный аргумент - df. Нужно предварительно подготовить df и перевести столбцы в категориальный 
    тип, есть подозрения на то, что выбросы как-то увязаны с этими столбцами. 
    Перебирает все числовые столбцы, для каждого строит боксплот, собирает информацию отдельно
    про верхние и про нижние выбросы (выводит таблицами под рисунком) Если в какой-либо категории выбросов 
    на 10+% больше, чем в среднем (и более 3% от общего числа), выводит информацию о ней на экран. Для 
    исключения столбцов типа category нужно передать их в кавычках через пробел в параметре exclude_cat=
    В параметрах include= и exclude= принимает названия столбцов, разделённые пробелом. Если параметры 
    не переданы, будет работать со всеми числовыми столбцами. Если передан include= , будет работать 
    только с теми столбцами, которые в include. От значений bot_whiskers= и top_whiskers= (пер. 'усы') 
    зависит то, какие значения будут считаться выбросами. По умолчанию все whiskers= 1.5 (полтора 
    межквартильных размаха), значения zero и unknown - -909090 и -777777.
    Рисунок, выполненный функцией, хорош тем, что верхние и нижние выбросы масштабируются
    по собственным осям Y, и ещё тем, что он даёт представление о том, как распределены значения не 
    только за пределами полутора (или сколько задано) межквартильных размахов, но и внутри. 
    Нужно иметь в виду, что пустое расстояние между столбцами сокращено. Гистограмма построена с помощью
    функции cut( ), начиная от значений Q1 и Q3, но при выбросах на значительную величину
    большая часть значений между Q1, Q3 и 1.5iqr попадает в первые несколько корзин. Визуально это
    выглядит так: всего несколько столбцов заходят внутрь боксплота. Несмотря на это неудобство, cut( ) 
    предпочтительнее чем qcut( ) в работе с редкими, случайно или ошибочно возникающими значениями. 
    С qcut получилось бы наоборот - почти все корзины пошли бы на регулярные значения, и всего несколько - 
    на выбросы.
    """
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'display' not in dir():
        from IPython.display import display
    import matplotlib.patheffects as path_effects
        
    style = dict(size = 11, color = '#929591',  
             path_effects = [path_effects.SimpleLineShadow(linewidth = 2, foreground = 'black'), 
                             path_effects.Normal()])
    counter = 0
    
    categories_list = list(df.select_dtypes(include = ['category']).columns)
    if len(exclude_cat) > 0:
        for x in exclude_cat.split():
            categories_list.remove(x)
    
    proceeding_list = list(
                           df.select_dtypes(include = ['uint8', 'uint16', 'uint32', 'uint64', 
                                                       'int8', 'int16', 'int32', 'int64', 
                                                       'float16', 'float32', 'float64']).columns)
    
    if len(exclude) > 0:
        for x in exclude.split():
            proceeding_list.remove(x)
    
    if len(include) > 0:
        proceeding_list = include.split()
        
    leng = len(proceeding_list) # for a counter
        
    outliers_table = pd.DataFrame(index = ['top', 'bottom'], 
                                  columns = ['total_rows', 'count', 'prc', 'min', 'max'])
    # outliers_tables' cells will be renewed in every iteration
        
    for col in proceeding_list:
        counter += 1
        print(f'Processing {counter} out of {leng}: {col}...')

        df_no_gaps = df[ (df[col] != unknown) & (df[col] != zero) & df[col].notna() ]
        total_rows = df_no_gaps.shape[0]   
    
        Q3 = df_no_gaps[col].quantile(0.75)
        Q2 = df_no_gaps[col].quantile(0.5)
        Q1 = df_no_gaps[col].quantile(0.25)
        iq_range = Q3 - Q1
        top_threshold = Q3 + (iq_range * top_whisker)
        bot_threshold = Q1 - (iq_range * bot_whisker)

        # making two slices containing all df columns (i'll need them to learn the way outskirts spread)
        df_top_slice = df_no_gaps[ df_no_gaps[col] > top_threshold ]
        df_bot_slice = df_no_gaps[ df_no_gaps[col] < bot_threshold ]
        
        if (df_top_slice.shape[0] == 0) and (df_bot_slice.shape[0] == 0):
            print(f"Outlying values in '{col}' column are not found.")
            print('Type anything to continue or type [exit]:', end = '')
            if input() == 'exit':
                return
            continue 
            
        # making two booleans: top_outliers_exist and bot_outliers_exist for future actions
        bot_outliers_exist = False
        top_outliers_exist = False
        if df_top_slice.shape[0] > 0:
            top_outliers_exist = True
        if df_bot_slice.shape[0] > 0:
            bot_outliers_exist = True

        # making two slices of a target column from min() to Q1 and from Q3 to max()
        if top_outliers_exist:
            right_slice = df_no_gaps[ df_no_gaps[col] > Q3 ][col]

        # creating top_value_counts and bot_value_counts (that's what we'll be plotting) via many 
        # functions combined
            top_value_counts = pd.cut(right_slice, 100, precision = 3, duplicates = 'drop').value_counts()
            top_value_counts = top_value_counts[ top_value_counts > 0 ].sort_index().reset_index()
            # due to 32b-Win i cant run some cut( )s when the number of bins exceeds 100. 
            # an error 'cannot cast array from dtype...' raises. so the numver is 100. 
            # if 100 is too much - 'duplicates=' parameter will work. if it works, i am
            # not concerned about losing steady progression. i lose it anyway after 
            # completing [... > 0] slice. i need to make this slice cause the distance between 
            # outlying values can be enormous - i have no need in illustrating it via
            # drawing all empty intervals on an ax.

        # receiving a closest to bot_threshold and top_threshold interval:
        # getting and index of the value that is the closest to a threshold and repeating
        # a cut( ) operation on a series, then receiving purposed interval via .loc[]
            closest_to_top_thr = right_slice.loc[ right_slice > top_threshold ].sort_values().iloc[0]
            top_thr_interval = (pd.cut(right_slice.append(pd.Series(closest_to_top_thr)), 
                                 100, precision = 3, duplicates = 'drop').iloc[-1])
            # append( ) operation doesn't change bins' intervals that are fabricated by cut( ) 
            # (comparing to '_value_counts'), cause the value we're appending is the same 
            # one that already exists a base slice

        # what is the counting number of an interval that contains a threshold? 
            top_thr_bin = list(top_value_counts.query('index == @top_thr_interval').index)[0]
            # a counting number is a number of bar that will represent an interval on a plot.
            # we need the counting number of an interval that includes a threshold value for three things:
            # we'll adjust boxplots' location using it as a reference point, all the bars before/after 
            # top/bot_thr_bin we'll make progressively transparent, and we'll use it as a key for slicing, 
            # like on the next step

        # receiving ylim for ax_top (1.2 of max occurence among outliers)
            top_ylim = top_value_counts.loc[top_thr_bin:, col].max() * 1.2
            # as you may've noticed, we've made one slice to Q1, and one from Q3, although the plot is 
            # dedicated to outliers, not to the values beyond Q1 and above Q3. That's why we need to 
            # explicitly set such ylim that would let us examine both outlying values and their relation 
            # to 'in-lying' values at the same time
            
        # searching for the closest (to the threshold) 'in-lier' value within iqr. if i set as xlim a 
        # 'theoretical' threshold, which results relatively far from the closest 'in-lier',  
        # a whisker of a boxplot will be shorter than axs' border
            closest_to_top_thr_from_inside = right_slice.loc[ right_slice <= top_threshold ].sort_values().iloc[-1]
        
        # counting the number of bins for top
            top_bins = top_value_counts.shape[0]
            
        # creating alpha_mask - list of an alpha parameter for every bar. the idea is to make bars 
        # more transparent while they are getting closer to Q3/Q1 
            alpha_mask_top = list(np.linspace(0, 1, top_thr_bin))
            alpha_mask_top.extend([1 for i in range(top_bins - top_thr_bin)])
            
        # if y-lim is six times higher than some bar, i'm afraid that we won't see the bar itself. i want to
        # add text descriptions to such bars. for this reason i'm making another column in '_value_counts'
            top_value_counts['needs_description'] = (top_value_counts[col] * 6 <= top_ylim)
            
        # finally i want to pick every forth bars' location as a place for a tick, starting from the last one.
        # i stop placing ticks when it is 3 bars to threshold
            tick_locations_top = [i for i in range(top_bins -1, top_thr_bin + 3, -4)]
            # there will be no error, just an empty list if the command is, e.g. '... in range(5, 12, -10)'
            # -1 cause the index of 54th element is 53
        
        else:
            top_bins = 1
            # i assign one bin to empty slices to reserve the possibility to create an ax

        # now we do all the same things to bottom outliers
        if bot_outliers_exist:
            left_slice = df_no_gaps[ df_no_gaps[col] < Q1 ][col]
            bot_value_counts = pd.cut(left_slice, 100, precision = 3, duplicates = 'drop').value_counts()
            bot_value_counts = bot_value_counts[ bot_value_counts > 0 ].sort_index().reset_index()
            closest_to_bot_thr = left_slice.loc[ left_slice < bot_threshold ].sort_values().iloc[-1]
            bot_thr_interval = (pd.cut(left_slice.append(pd.Series(closest_to_bot_thr)), 
                                 100, precision = 3, duplicates = 'drop').iloc[-1])
            bot_thr_bin = list(bot_value_counts.query('index == @bot_thr_interval').index)[0]
            bot_ylim = bot_value_counts.loc[:bot_thr_bin, col].max() * 1.2
            closest_to_bot_thr_from_inside = left_slice.loc[ left_slice >= bot_threshold ].sort_values().iloc[0]
            bot_bins = bot_value_counts.shape[0]
            alpha_mask_bot = [1 for i in range(bot_thr_bin +1)]
            alpha_mask_bot.extend(list(np.linspace(1, 0, (bot_bins - bot_thr_bin))))
            bot_value_counts['needs_description'] = (bot_value_counts[col] * 6 <= bot_ylim)
            tick_locations_bot = [i for i in range(0, bot_thr_bin -3, 4)]
        else:
            bot_bins = 1

        # counting the number of bins for center
        center_bins = (pd.cut(
            df_no_gaps.loc[ (df_no_gaps[col] >= Q1) & (df_no_gaps[col] <= Q3), col], 50, duplicates = 'drop' )
            .cat.categories.shape[0])
            # bins = 50 means (while edges had 100) that ideally i want to have a 1:2 proportion boxplot:edges

        # counting the total number of bins on picture and the width of a picture (= of a bin)
        bins_total = center_bins + top_bins + bot_bins
        figure_width = 0.13 * bins_total
        if figure_width > 18:
            figure_width = 18
        if figure_width < 8:
            figure_width = 8
        
        
        """
        # DEBUG PACK:
        
        print(bins_total)
        print(center_bins, 'center')
        print(top_bins, 'top')
        print(bot_bins, 'bot')
        print('\n' * 2)
        
        if bot_outliers_exist:
            print('BOTTOM')
            print('\n')
            print(bot_value_counts)
            print('\n')
            print('ax_bot = fig.add_subplot(grid[:, :bot_bins])')
            print('\n')
            print(bot_threshold, 'bot threshold')
            print(bot_thr_bin, 'bot threshold bin')
            print(len(alpha_mask_bot), 'alpha_mask_lengh bot')
            print(tick_locations_bot, 'tick locations bot')
            print('\n' * 2)
        
        if top_outliers_exist:
            print('TOP')
            print('\n')
            print(top_value_counts)
            print('\n')
            print('ax_top = fig.add_subplot(grid[:, -(top_bins):])')
            print('\n')
            print(top_threshold, 'top threshold')
            print(top_thr_bin, 'top threshold bin')
            print(len(alpha_mask_top), 'alpha_mask_lengh bot')
            print(tick_locations_top, 'tick locations top')
            print('\n' * 2)
        """
            
        # creating a column with customized spelling of intervals
        def customizing_intervals(interval):
            interval = str(interval)
            left, right = interval.split()
            left_integer, left_decimals = left.split('.')
            left_integer = left_integer[1:]
            left_decimals = left_decimals[0]
            right_integer, right_decimals = right.split('.')
            right_decimals = right_decimals[0]
            return ''.join([left_integer, '.', left_decimals, ' - ', right_integer, '.', right_decimals])
        if bot_outliers_exist:
            bot_value_counts['labels'] = bot_value_counts['index'].apply(customizing_intervals)
        if top_outliers_exist:
            top_value_counts['labels'] = top_value_counts['index'].apply(customizing_intervals)
          
        # creating a figure and an empty_ax that will fulfill an emtpy space. disabling all ticks
        # creating grid with 5 rows (boxplot will be placed on two lowests) and as many rows as there will be bins
        fig = plt.figure(figsize = (figure_width, 5))
        grid = fig.add_gridspec(6, bins_total, wspace = 0, hspace = 0)
        empty_ax = fig.add_subplot(grid[:5, :])
        empty_ax.tick_params(labelleft = False, labelbottom = False, grid_alpha = 0)
        
        # creating bars: all the code below is just one conditional expression with one fragment pasted 
        # three times, so don't be jaded if you are
        if top_outliers_exist and bot_outliers_exist:
            fig.suptitle(f'{col}: cut({bot_bins}) from min to Q1, cut({top_bins}) from Q3 to max', fontsize = 15)
            ax_bot = fig.add_subplot(grid[:5, :bot_bins])
            ax_bot.tick_params(grid_alpha = 0)
            ax_bot.set(xlim = (0, bot_bins), ylim = (0, bot_ylim))
            for i in list(bot_value_counts.index):
                ax_bot.bar(i, bot_value_counts.loc[i, col], width = 0.6, color = '#ff000d', alpha = alpha_mask_bot[i])
                if bot_value_counts.loc[i, 'needs_description']:
                    ax_bot.text(i, bot_value_counts.loc[i, col], bot_value_counts.loc[i, col], **style)
            ax_bot.set_xticks(tick_locations_bot)
            ax_bot.set_xticklabels([bot_value_counts.loc[i, 'labels'] for i in tick_locations_bot], rotation = 60)
            ax_bot.spines['right'].set_visible(False)

            ax_top = fig.add_subplot(grid[:5, -(top_bins):])
            ax_top.tick_params(labelright = True, labelleft = False, grid_alpha = 0)               
            ax_top.set(xlim = (0, top_bins), ylim = (0, top_ylim))
            for i in list(top_value_counts.index):
                ax_top.bar(i, top_value_counts.loc[i, col], width = 0.6, color = '#0504aa', alpha = alpha_mask_top[i])
                if top_value_counts.loc[i, 'needs_description']:
                    ax_top.text(i, top_value_counts.loc[i, col], top_value_counts.loc[i, col], **style)
            ax_top.set_xticks(tick_locations_top)
            ax_top.set_xticklabels([top_value_counts.loc[i, 'labels'] for i in tick_locations_top], rotation = 60)
            ax_top.spines['left'].set_visible(False)
            
       # creating central_ax in the the bottom of the figure. creating a boxplot over a definite slice. 
       # setting ylim from 0.9 to 1.1 to widen the box of boxplot, setting xlim and xticks
            ax_central = fig.add_subplot(grid[4:6, (bot_thr_bin +1):-(top_bins - top_thr_bin +1)])
            ax_central.set(xlim = (closest_to_bot_thr_from_inside, closest_to_top_thr_from_inside), ylim = (0.8, 1.2))
            ax_central.tick_params(labelleft = False, labelbottom = True, grid_alpha = 0)
            ax_central.set_xticks([closest_to_bot_thr_from_inside, Q1, Q2, Q3, closest_to_top_thr_from_inside])
            ax_central.set_xticklabels(['{:.1f}'.format(bot_threshold), '{:.1f}'.format(Q1), 
                                        '{:.1f}'.format(Q2), '{:.1f}'.format(Q3), '{:.1f}'.format(top_threshold)])
            
        
        elif top_outliers_exist:
            fig.suptitle(f'{col}: cut({top_bins}) from Q3 to max', fontsize = 15)
            empty_ax.tick_params(labelleft = False)
            empty_ax.set(ylabel = 'No bottom outliers')
            ax_top = fig.add_subplot(grid[:5, -(top_bins):])
            ax_top.tick_params(labelright = True, labelleft = False, grid_alpha = 0)               
            ax_top.set(xlim = (0, top_bins), ylim = (0, top_ylim))      
            for i in list(top_value_counts.index):
                ax_top.bar(i, top_value_counts.loc[i, col], width = 0.6, color = '#0504aa', alpha = alpha_mask_top[i])
                if top_value_counts.loc[i, 'needs_description']:
                    ax_top.text(i, top_value_counts.loc[i, col], top_value_counts.loc[i, col], **style)
            ax_top.set_xticks(tick_locations_top)
            ax_top.set_xticklabels([top_value_counts.loc[i, 'labels'] for i in tick_locations_top], rotation = 60)
            ax_top.spines['left'].set_visible(False)
            
            ax_central = fig.add_subplot(grid[4:6, :-(top_bins - top_thr_bin)])
            ax_central.set(xlim = (df_no_gaps[col].min(), closest_to_top_thr_from_inside), ylim = (0.8, 1.2))
            ax_central.tick_params(labelleft = False, labelbottom = True, grid_alpha = 0)
            ax_central.set_xticks([df_no_gaps[col].min(), Q1, Q2, Q3, closest_to_top_thr_from_inside])
            ax_central.set_xticklabels(['{:.1f}'.format(df_no_gaps[col].min()), '{:.1f}'.format(Q1), 
                                        '{:.1f}'.format(Q2), '{:.1f}'.format(Q3), '{:.1f}'.format(top_threshold)])
            

        elif bot_outliers_exist:
            fig.suptitle(f'{col}: cut({bot_bins}) from min to Q1', fontsize = 15)             
            empty_ax.tick_params(labelleft = False)
            empty_ax.yaxis.set_label_position('right')
            empty_ax.set_ylabel('No top outliers')
            ax_bot = fig.add_subplot(grid[:5, :bot_bins])
            ax_bot.tick_params(grid_alpha = 0)
            ax_bot.set(xlim = (0, bot_bins), ylim = (0, bot_ylim))
            for i in list(bot_value_counts.index):
                ax_bot.bar(i, bot_value_counts.loc[i, col], width = 0.6, color = '#ff000d', alpha = alpha_mask_bot[i])
                if bot_value_counts.loc[i, 'needs_description']:
                    ax_bot.text(i, bot_value_counts.loc[i, col], bot_value_counts.loc[i, col], **style)
            ax_bot.set_xticks(tick_locations_bot)
            ax_bot.set_xticklabels([bot_value_counts.loc[i, 'labels'] for i in tick_locations_bot], rotation = 60)
            ax_bot.spines['right'].set_visible(False)
            
            ax_central = fig.add_subplot(grid[4:6, (bot_thr_bin +1):])
            ax_central.set(xlim = (closest_to_bot_thr_from_inside, df_no_gaps[col].max()), ylim = (0.8, 1.2))
            ax_central.tick_params(labelleft = False, labelbottom = True, grid_alpha = 0)
            ax_central.set_xticks([closest_to_bot_thr_from_inside, Q1, Q2, Q3, df_no_gaps[col].max()])
            ax_central.set_xticklabels(['{:.1f}'.format(bot_threshold), '{:.1f}'.format(Q1), 
                                        '{:.1f}'.format(Q2), '{:.1f}'.format(Q3), '{:.1f}'.format(df_no_gaps[col].max())])

           
        # creating and customizing boxplot     
        bp = ax_central.boxplot(df_no_gaps[col], 
                        vert = False, patch_artist = True, showfliers = False, 
                        whis = max([bot_whisker, top_whisker]), notch = True)
        for whiskers in bp['whiskers']:
            whiskers = whiskers.set(color = '#ad8150', linewidth = 10)
        for caps in bp['caps']:
            caps = caps.set(color = '#ad8150', linewidth = 10)
        for medians in bp['medians']:
            medians = medians.set(color = 'red', linewidth = 2)
        for boxes in bp['boxes']:
            boxes = boxes.set_facecolor('grey')
            
        display(fig)
        plt.close(fig)
        

        outliers_table.loc[:, 'total_rows'] = total_rows
        outliers_table.loc['top', 'count'] = df_top_slice.shape[0]
        outliers_table.loc['bottom', 'count'] = df_bot_slice.shape[0]
        outliers_table.loc['top', 'prc'] = '{:.2%}'.format(outliers_table.loc['top', 'count'] / total_rows)
        outliers_table.loc['bottom', 'prc'] = '{:.2%}'.format(outliers_table.loc['bottom', 'count'] / total_rows)
        outliers_table.loc['top', 'min'] = df_top_slice[col].min()
        outliers_table.loc['bottom', 'min'] = df_bot_slice[col].min()
        outliers_table.loc['top', 'max'] = df_top_slice[col].max()
        outliers_table.loc['bottom', 'max'] = df_bot_slice[col].max()
        display(outliers_table.fillna('-'))
        print('\n')
    
        for cat_col in categories_list:
            comparing_categories = pd.concat([df_no_gaps[cat_col].value_counts(),
                                              df_top_slice[cat_col].value_counts(),
                                              df_top_slice[cat_col].value_counts(normalize = True),
                                              # top_outliers in each_category
                                              df_bot_slice[cat_col].value_counts(),
                                              df_bot_slice[cat_col].value_counts(normalize = True)],
                                              # bot_outliers in each_category
                                              axis = 1)
            comparing_categories.columns = ['values_in_category', 'top_outliers', 'top_outliers_prc', 
                                            'bot_outliers', 'bot_outliers_prc']
            comparing_categories['consists_of_top_outliers'] = (comparing_categories['top_outliers']
                                                                .div(comparing_categories['values_in_category'])
                                                                .fillna(0))
            comparing_categories['consists_of_bot_outliers'] = (comparing_categories['bot_outliers']
                                                                .div(comparing_categories['values_in_category'])
                                                                .fillna(0))
            # div( ) method supports division by zero, 
            # i can use it confidently even if a column has unused categories with 0 in 'value_counts'
            
            
            top_threshold_cat = comparing_categories['top_outliers'].mean() * 1.1
            bot_threshold_cat = comparing_categories['bot_outliers'].mean() * 1.1
            # all categories are included to .value_counts( ) resulting dataframe, even
            # if their count is 0, so .mean( ) will represent the situation correctly.
            # 0.1 (10%) is an acceptable fluctuation within a category. if more, i
            # want to be informed about this category
            
            remarkable_cats_top = (comparing_categories.loc[ 
                                                (comparing_categories['top_outliers'] > top_threshold_cat) &
                                                (comparing_categories['top_outliers_prc'] > 0.03)].index)
            remarkable_cats_bot = (comparing_categories.loc[ 
                                                (comparing_categories['bot_outliers'] > bot_threshold_cat) &
                                                (comparing_categories['bot_outliers_prc'] > 0.03)].index)
            homogeneous_cats_top = (comparing_categories.loc[ 
                                                comparing_categories['consists_of_top_outliers'] > 0.5].index)
            homogeneous_cats_bot = (comparing_categories.loc[ 
                                                comparing_categories['consists_of_bot_outliers'] > 0.5].index)
            # i want to be informed if any category consists of the rows 
            # with top or bot outliers more more than 0.5 (50%) percent
            
            if ((len(remarkable_cats_top) == 0) and (len(remarkable_cats_bot) == 0) and 
                (len(homogeneous_cats_top) == 0) and (len(homogeneous_cats_bot) == 0)):
                print(f"There is nothing special about '{col}' outliers in '{cat_col}' column.")
                continue
                
            top_mean_for_ordinary_cats = (comparing_categories.loc[ ~comparing_categories
                                                                  .index.isin(remarkable_cats_top),
                                                                  'top_outliers_prc']).mean()
            bot_mean_for_ordinary_cats = (comparing_categories.loc[ ~comparing_categories
                                                                  .index.isin(remarkable_cats_bot),
                                                                  'bot_outliers_prc']).mean()
            # i need these variables just to make an output that i will be able to understand
            
            if len(remarkable_cats_top) > 0:
                for remarkable_cat in remarkable_cats_top:
                    print('{: <25}'.format(cat_col), 
                          '> {: <25}'.format('{}'.format(remarkable_cat)), 
                          ' {:.1%}'.format(comparing_categories.loc[remarkable_cat, 'top_outliers_prc']), 
                          ' of all TOP outliers', sep = '')
                number_of_others = comparing_categories.shape[0] - len(remarkable_cats_top)
                print('{: <25}'.format(cat_col), 
                          '> {: <24}'.format(f'other categories ({number_of_others})'), 
                          ' ~{:.1%}'.format(top_mean_for_ordinary_cats), 
                          ' of all TOP outliers in each one', sep = '')
                print()
            if len(homogeneous_cats_top) > 0:
                for homogeneous_cat in homogeneous_cats_top:
                    print('{: <25}'.format(cat_col), 
                      '> {: <25}'.format('{}'.format(homogeneous_cat)), 
                      ' {:.1%}'.format(comparing_categories.loc[homogeneous_cat, 'consists_of_top_outliers']), 
                      f' of rows are TOP outliers', 
                      sep = '')
                print()
            else:
                print()
            
            if len(remarkable_cats_bot) > 0:
                for remarkable_cat in remarkable_cats_bot:
                    print('{: <25}'.format(cat_col), 
                          '> {: <25}'.format('{}'.format(remarkable_cat)), 
                          ' {:.1%}'.format(comparing_categories.loc[remarkable_cat, 'bot_outliers_prc']), 
                          ' of all BOTTOM outliers', sep = '')
                    number_of_others = comparing_categories.shape[0] - len(remarkable_cats_bot)
                print('{: <25}'.format(cat_col), 
                          '> {: <24}'.format(f'other categories ({number_of_others})'), 
                          ' ~{:.1%}'.format(bot_mean_for_ordinary_cats), 
                          ' of all BOTTOM outliers in each one', sep = '')
                print()
            if len(homogeneous_cats_bot) > 0:
                for homogeneous_cat in homogeneous_cats_bot:
                    print('{: <25}'.format(cat_col), 
                      '> {: <25}'.format('{}'.format(homogeneous_cat)), 
                      ' {:.1%}'.format(comparing_categories.loc[homogeneous_cat, 'consists_of_bot_outliers']), 
                      f' of rows are BOTTOM outliers', 
                      sep = '')
                    
            print('{:_<100}'.format(''))
            print()
                
                
                
        incorrect = True
        print('Make a decision. Enter [all / top .* / bot .* / n / exit]:', end = ' ')
        while incorrect:
            command = input()
            if len(command.split()) == 2:
                try:
                    key = float(command.split()[1])
                    key_is_given = True
                except ValueError:
                    print("Can't convert your key to float type. Try again:", end = ' ')
                    continue
            else:
                key_is_given = False

            if command == 'all':
                before = df[ ((df[col] <= bot_threshold) | (df[col] >= top_threshold)) & (df[col].notna()) 
                             & (df[col] != zero) & (df[col] != unknown) ].shape[0]
                df[col] = df[col].where( (df[col].isna()) | (df[col] == zero) | (df[col] == unknown) | 
                                               (df[col] > bot_threshold) | (df[col] < top_threshold), unknown )
                if before != 0:
                    incorrect = False
                    print(f"{before} values in '{col}' were replaced by '{unknown}'")
                else: 
                    print("There is no sence in removing data from an empty slice. Try again:", end = ' ')
            elif command == 'top':
                before = df[ (df[col] >= top_threshold) & (df[col].notna()) & (df[col] != zero) & 
                                    (df[col] != unknown) ].shape[0]
                df[col] = df[col].where( (df[col].isna()) | (df[col] == zero) | (df[col] == unknown) | 
                                               (df[col] < top_threshold), unknown )
                if before != 0:
                    incorrect = False
                    print(f"{before} top-outliers in '{col}' were replaced by '{unknown}'")
                else: 
                    print("There is no sence in removing data from an empty slice. Try again:", end = ' ')
            elif command == 'bot':
                before = df[ (df[col] <= bot_threshold) & (df[col].notna()) & (df[col] != zero) & 
                                    (df[col] != unknown) ].shape[0]
                df[col] = df[col].where( (df[col].isna()) | (df[col] == zero) | (df[col] == unknown) | 
                                               (df[col] > bot_threshold), unknown )
                if before != 0:
                    incorrect = False
                    print(f"{before} bot-outliers in '{col}' were replaced by '{unknown}'")
                else: 
                    print("There is no sence in removing data from an empty slice. Try again:", end = ' ')
            elif command == 'n':
                incorrect = False
                pass
            elif command == 'exit':
                return
            elif key_is_given:
                if command.split()[0] == 'top':
                    try:
                        before = df[ (df[col] >= key) & (df[col].notna()) & (df[col] != zero) & 
                                    (df[col] != unknown) ].shape[0]
                        df[col] = df[col].where( (df[col].isna()) | (df[col] == zero) | 
                                                       (df[col] == unknown) | (df[col] < key), unknown )
                        if before != 0:
                            incorrect = False
                            print(f"{before} values equal or greater than {key} were replaced by '{unknown}' in '{col}'")
                        else: 
                            print("You've probably passed a wrong key: there is no sence in removing data from an empty slice. Try again:", end = ' ')
                    except TypeError:
                        print("You've passed an invalid key - TypeError has arised. Try again:", end = ' ')
                elif command.split()[0] == 'bot':
                    try:
                        before = df[ (df[col] <= key) & (df[col].notna()) & (df[col] != zero) & 
                                    (df[col] != unknown) ].shape[0]
                        df[col] = df[col].where( (df[col].isna()) | (df[col] == zero) | 
                                                       (df[col] == unknown) | (df[col] > key), unknown )
                        if before != 0:
                            incorrect = False
                            print(f"{before} values equal or less than {key} were replaced by '{unknown}' in '{col}'")
                        else: 
                            print("You've probably passed a wrong key: there is no sence in removing data from an empty slice. Try again:", end = ' ')
                    except TypeError:
                        print("You've passed an invalid key - TypeError has arised. Try again:", end = ' ')
                else:
                    print("Seems like you've tried to pass a key, but haven't mentioned whether you want to remove 'bot' or 'top' values. Try again:", end = ' ')
            else: 
                print('Incorrect input. Try again:', end = ' ')
        
    return
#%%
def dumbbells(df, Y_col, X_col, Z_col, before = '', after = '', func = '', title = '',
              particularly = True, dropna = False, drop_zero = False, fillna_zero = True):
    """
    Строит график 'гантели'. Первый аргумент - датафрейм, второй - столбец, для уникальных значений
    которого будет построен график (ось Y). Третий аргумент - столбец со значениями числовой 
    переменной , ось Х. Четвёртый аргумент: столбец, влияние которого на уникальные значения и 
    иллюстрируется графиком. Его значения "до" и "после" передаются через аргументы before= 
    и after=. Если его значения числовые, то "before" и "after" можно сделать двумя корзинами
    (все значения меньше чем before и все значения больше чем after), передав particularly = False.
    Если dropna = True, то будут удалены те уникальные значения Y, для которых нет одного из 
    значений before или after (NaN). По умолчанию dropna = False. Можно также поменять нули
    на пропуски (когда они и в before, и в after) и удалить их, передав drop_zero = True (по 
    умолчанию False). В итоге drop_zero убирает значения, которым соответствует ноль и "до", и "после", 
    а dropna убирает значения, которым хотя бы раз соответствует пропуск. 'func=' указывает, какую 
    функцию применять к значениям X после группировки по значениям Y: 'count', 'median', 'sum' или 'mean'.
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    if not 'display' in dir():
        from IPython.display import display
        
    
    if particularly:
        legend = [':', ':']
        if func == 'sum':
            GB_before = df[ df[Z_col] == before ].groupby(Y_col)[X_col].sum()
            GB_after = df[ df[Z_col] == after ].groupby(Y_col)[X_col].sum()
            label = 'сумма значений '
        elif func == 'median':
            GB_before = df[ df[Z_col] == before ].groupby(Y_col)[X_col].median()
            GB_after = df[ df[Z_col] == after ].groupby(Y_col)[X_col].median()
            label = 'медиана значений '
        elif func == 'mean':
            GB_before = df[ df[Z_col] == before ].groupby(Y_col)[X_col].mean()
            GB_after = df[ df[Z_col] == after ].groupby(Y_col)[X_col].mean()
            label = 'среднее значений столбца '
        elif func == 'count':
            GB_before = df[ df[Z_col] == before ].groupby(Y_col)[X_col].count()
            GB_after = df[ df[Z_col] == after ].groupby(Y_col)[X_col].count()
            label = 'количество значений столбца '
        else:
            print("Введи параметр func = 'sum' / 'mean' / 'median' / 'count'")
            return
    else:
        legend = ['before:', 'after:']
        if func == 'sum':
            GB_before = df[ df[Z_col] < before ].groupby(Y_col)[X_col].sum()
            GB_after = df[ df[Z_col] > after ].groupby(Y_col)[X_col].sum()
            label = 'сумма значений '
        elif func == 'median':
            GB_before = df[ df[Z_col] < before ].groupby(Y_col)[X_col].median()
            GB_after = df[ df[Z_col] > after ].groupby(Y_col)[X_col].median()
            label = 'медиана значений '
        elif func == 'mean':
            GB_before = df[ df[Z_col] < before ].groupby(Y_col)[X_col].mean()
            GB_after = df[ df[Z_col] > after ].groupby(Y_col)[X_col].mean()
            label = 'среднее значений столбца '
        elif func == 'count':
            GB_before = df[ df[Z_col] < before ].groupby(Y_col)[X_col].count()
            GB_after = df[ df[Z_col] > after ].groupby(Y_col)[X_col].count()
            label = 'количество значений столбца '
        else:
            print("Введи параметр func = 'sum' / 'mean' / 'median' / 'count'")
            return

    GB_united = pd.concat([GB_before, GB_after], axis = 1).set_axis(['before', 'after'], axis = 1)  

    GB_united = GB_united.sort_values(['after', 'before'])
    if drop_zero:
        GB_united.drop(GB_united[ (GB_united['before'] == 0) & (GB_united['after'] == 0) ].index, 
                       axis = 0, inplace = True)
    if dropna:
        GB_united = GB_united.dropna()
    else:
        GB_united = GB_united.dropna(how = 'all', axis = 0)
    
    uniques = GB_united.index.to_list()
    uniques_length = len(uniques)
    fig, ax = plt.subplots(figsize = (12, uniques_length * 0.5 + 1))
    ax.set(ylim = (-1, uniques_length), yticks = np.arange(uniques_length), yticklabels = uniques,
           xlabel = label + "'" + X_col + "'")
    ax.set_title(title, fontsize = 20)
    ax.grid(b = False, axis = 'both')
    ax.hlines(y = ax.get_yticks(), xmin = GB_united.min().min() * 0.95, 
              xmax = GB_united.max().max() * 1.05, color = 'gray',
              alpha = 0.5, linewidth = 2, linestyles = 'dotted')
    
    for i, name in enumerate(uniques):
        bef = GB_united.loc[name, 'before']
        aft = GB_united.loc[name, 'after']
        ax.scatter(y = [i], x = bef, marker = 'x', c = '#c69f59', linewidths = 25, 
                   edgecolors = '#c69f59')
        ax.scatter(y = [i], x = aft, marker = 'o', c = '#0bf9ea', linewidths = 25, 
                   edgecolors = '#0bf9ea')
        if np.isnan(aft) == False and np.isnan(aft) == False:
            ax.hlines(xmin = bef, y = i, xmax = aft, linewidth = 6, 
                      color = ['#c69f59' if bef > aft else '#0bf9ea'][0], alpha = 0.4)
    
    ax.scatter([], [], marker = 'x', c = '#c69f59', linewidths = 25, 
                   edgecolors = '#c69f59', label = legend[0] + f' {before}')
    ax.scatter([], [], marker = 'o', c = '#0bf9ea', linewidths = 25, 
                   edgecolors = '#0bf9ea', label = legend[1] + f' {after}')
    ax.legend(loc = 'lower right', fontsize = 'xx-large', ncol = 2)
    
    display(fig)
    plt.close(fig)
    return
#%%
def std_ddof(collection_of_samples):
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    
    if len(collection_of_samples) == 1:
        std = np.std(collection_of_samples.astype('float'), ddof = 1)
        print(round(std, 3))
    else:
        all_vars = []
        for sample in collection_of_samples:
            std = np.std(sample.astype('float'), ddof = 1)
            all_vars.append(np.var(sample.astype('float'), ddof = 1))
            print(round(std, 3), end = ' ')
        minimal = min(all_vars); maximal = max(all_vars)
        if minimal * 1.05 < maximal:
            print(f"\nДисперсии ({round(minimal, 2)} и {round(maximal, 2)}) отличаются более чем на 5%")
        else:
            print("\nДисперсии отличаются менее чем на 5%")
#%%     
def pie_chart(df, col, dec = 0, cmap = 'tab20b', title = '', reduce_to = [], ax = False):
    """
    Рисует круговую диаграмму. Первый аргумент - датафрейм, второй - столбец, уникальные 
    значения которого будут отображены на диаграмме. Посредством параметра 'reduce_to =' 
    можно передать список категорий, которые отобразятся на диаграмме (остальные
    будут объеденены под именем 'Others'). 'title =' - заголовок, 'dec =' - сколько знаков
    после запятой будет отображаться рядом с категориями (по умолчанию - 0), 'cmap =' -
    какая цветовая палитра будет использована (по умолчанию 'tab20b').
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'pd' not in dir():
        import pandas as pd
    if not 'display' in dir():
        from IPython.display import display
        
    dec = '{:.' + str(dec) + '%}'
        
    if len(reduce_to) != 0:
        reduce_to = list(reduce_to)
        for_pie = df[col].astype('object').where(df[col].isin(reduce_to), 'Other').value_counts()
        for_pie.index = for_pie.index.astype('str')
        for_pie = for_pie.sort_index()
        labels = (df[col].astype('object').where(df[col].isin(reduce_to), 'Other')
                  .value_counts(normalize = True))
        labels.index = labels.index.astype('str')
        labels = labels.sort_index().apply(dec.format)
    else:
        for_pie = df.loc[:, col].astype('object').value_counts()
        for_pie.index = for_pie.index.astype('str')
        for_pie = for_pie.sort_index()
        labels = df.loc[:, col].astype('object').value_counts(normalize = True)
        labels.index = labels.index.astype('str')
        labels = labels.sort_index().apply(dec.format)
        
    leng = for_pie.shape[0]
    cmap = plt.get_cmap(cmap, leng)
    labels = (labels + ': ' + labels.index).to_list()

    if ax == False:
        fig, ax = plt.subplots(figsize = (5, 5))
        no_initial_ax = True
    else:
        no_initial_ax = False
    ax.set_title(title, fontsize = 15)
    wedges, texts = ax.pie(for_pie, wedgeprops = {'width': 0.5}, labels = labels, colors = cmap(range(leng)))
    
    if no_initial_ax:                       
        display(fig)
        plt.close(fig)
#%%
def boxplot_sequence(df, Y_column, X_column, categories = [], title = '', x_dec = 1, ax = False):
    """
    Строит боксплоты для нескольких категорий. Первый аргумент - датафрейм, второй - столбец с
    категориями (ось У), третий - столбец с изучаемой переменной (ось Х). Далее параметры: 
    `categories = ` принимает коллекцию категорий, для которых будут построены боксплоты, по 
    умолчанию - строит для всех. 'title = ' принимает строчку с заголовком, 'x_dec = ' 
    принимает количество цифр после запятой на отметках оси X. Выбросы не рисуются. Вместо этого при
    наличии выбросов указывается процент - доля выбросов от всех значений переменной внутри категории.
    Точка обозначает среднее. Если точка пурпурного цвета, значит, среднее вышло за 1.5 iqr. 
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    if 'display' not in dir():
        from IPython.display import display
    from warnings import filterwarnings
    filterwarnings("ignore")
        
    if hasattr(df[Y_column], 'cat'):
        df = df.copy()
        df[Y_column] = df[Y_column].astype('object')
        # понадобится сделать группировку, из которой нужно 
        # исключить все невостребованные категории

    if len(categories) == 0:
        categories = list(df[Y_column].dropna().unique())
    else:
        categories = list(categories)
    
    categories = (df[ df[Y_column].isin(categories) ].groupby(Y_column)[X_column].median()
                                                     .sort_values().index.to_list())
    leng = len(categories)
    thr_s = []
    
    if ax == False:
        fig, ax = plt.subplots(figsize = (12, leng * 0.8))
        no_initial_ax = True
    else:
        no_initial_ax = False
    ax.set_title(title, fontsize = 16, y = 1.03)
    ax.grid(b = False, axis = 'y')
    ax.grid(axis = 'x', alpha = 0.2, color = 'grey', zorder = 1)
    
    bps = ax.boxplot([df.loc[ df[Y_column] == cat, X_column ] for cat in categories], zorder = 2,
                     showfliers = False, vert = False, patch_artist = True)
    xlim_min, xlim_max = ax.get_xlim()
    ylim = ax.get_ylim()
    yticks = ax.get_yticks()
    ax.set(ylim = (ylim[0] - 0.4, ylim[1] + 0.4), yticks = yticks, 
           yticklabels = categories, xlabel = X_column)
    for whiskers in bps['whiskers']:
        whiskers = whiskers.set(color = '#dfc5fe', linewidth = 3)
    for caps in bps['caps']:
        caps = caps.set(color = '#856798', linewidth = 4)
    for medians in bps['medians']:
        medians = medians.set(color = '#fdff38', linewidth = 3)
    for boxes in bps['boxes']:
        boxes = boxes.set(facecolor = '#856798')
        
    for y, cat in zip(yticks, categories):
        sliced = df.loc[ (df[Y_column] == cat), X_column]
        sl_leng = sliced.shape[0]
        Q3 = sliced.quantile(0.75)
        Q1 = sliced.quantile(0.25)
        iq_range = Q3 - Q1
        top_threshold = Q3 + (iq_range * 1.5)
        bot_threshold = Q1 - (iq_range * 1.5)
        if top_threshold > sliced.max():
            top_threshold = sliced.max()
        if bot_threshold < sliced.min():
            bot_threshold = sliced.min()
        
        shift_magn = (xlim_max - xlim_min) * 0.02
        
        outl_top_leng = sliced[ sliced > top_threshold ].shape[0]
        outl_bot_leng = sliced[ sliced < bot_threshold ].shape[0]
        
        closest_to_top = sliced[ sliced < top_threshold ].max()
        closest_to_bot = sliced[ sliced > bot_threshold ].min()
        thr_s.append(closest_to_top)
        thr_s.append(closest_to_bot)
        
        mean = sliced.mean()
        color = '#dfc5fe'
        if mean > top_threshold:
            mean = closest_to_top
            color = '#c20078'
        elif mean < bot_threshold:
            mean = closest_to_bot
            color = '#c20078'
            
        ax.scatter(y = y, x = mean, c = color, s = 70, zorder = 3, edgecolors = 'grey')
        
        for out_number, if_bot in zip([outl_bot_leng, outl_top_leng], [True, False]):
            if out_number > 0:
                inscr = '{:.2%}'.format(out_number / sl_leng)
                
                if if_bot:
                    loc = closest_to_bot - shift_magn
                    ax.text(loc, y, inscr, color = 'grey', fontsize = 13.5, va = 'center', ha = 'right')
                    if (loc - xlim_min) < ((xlim_max - xlim_min) * 0.07):
                        xlim_min = xlim_min - ((xlim_max - xlim_min) * 0.07)
                        ax.set_xlim(xmin = xlim_min)
                else:
                    loc = closest_to_top + shift_magn
                    ax.text(loc, y, inscr, color = 'grey', fontsize = 13.5, va = 'center', ha = 'left')
                    if (xlim_max - loc) < ((xlim_max - xlim_min) * 0.08):
                        xlim_max = xlim_max + ((xlim_max - xlim_min) * 0.08)
                        ax.set_xlim(xmax = xlim_max)
                        
    x_ticks = np.linspace(min(thr_s), max(thr_s), 13)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([round(x, x_dec) if (x_dec != 0) else int(round(x, x_dec)) for x in x_ticks], rotation = 70)
    
    if no_initial_ax:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        display(fig)
        plt.close(fig)
#%%
def paired_ttest(first, second, alpha = 0.05, illustrate = True, descr = ['', ''], names = ['', ''], 
                 col_name = '', x_dec = 0):
    """
    Проверяет гомогенность дисперсий с помощью теста Левина. Проводит парный ttest, рисует 
    график распределения разностей и доверительные интервалы (если не передано illustrate = False).
    В параметрах 'descr =' и 'names =' принимает списки из двух строчек, соответствующих двум выборкам.
    В 'descr =' описание генеральных совокупностей, в 'names =' короткие названия выборок для легенды.
    В 'col_name =' принимает название столбца, содержащего передаваемые значения (для подписи оси Х в
    графике с доверительными интервалами). В параметре 'alpha =' принимает уровень значимости, по
    умолчанию - 0.05. В параметре 'dec =' принимает количество знаков после запятой в отметках на оси Х
    в графике распределения разностей. Первые два позиционных аргумента - первая и вторая выборки.
    Если разности превышают три стандартных ошибки, то соответствующие им две точки на графике окрашиваются
    в красный цвет и приподнимаются по оси У. Две точки - это две разности, смотря по тому, какое из двух
    средних из какого вычитается.
    """
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    if 'st' not in dir():
        import scipy.stats as st
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'display' not in dir():
        from IPython.display import display
    
    if (not isinstance(descr, list)) or (len(descr) != 2) or (not all([isinstance(x, str) for x in descr])):
        print("Параметр 'descr =' принимает список из двух строчек. Идея - описать два свойства")
        return 
    
    first = np.array(first)
    first = first[~ np.isnan(first)]
    second = np.array(second)
    second = second[~ np.isnan(second)]

    levene = st.levene(first, second).pvalue 
    print('Существуют две ген.совокупности:', f' 1. {descr[0]}', f' 2. {descr[1]}', sep = '\n', end = '\n\n')
    if levene < 0.05:
        equal_var = False
        print('Равенство их дисперсий не подтвердилось (тест Левена), equal_var == False', end = '\n\n')
    else:
        equal_var = True
        print('pvalue равенства их дисперсий равно {:.3f} (тест Левена), equal_var == True'.format(levene), 
              end = '\n\n')
    
    print("H0: Средние ген.совокупностей равны")
    result = st.ttest_ind(first, second, equal_var = equal_var)
    print('pvalue данных выборок или выборок с большим различием выборочных средних при допущении H0:', 
          str(round(result.pvalue, 4)))
    [print('\t- Отклоняем H0') if result.pvalue < alpha else print('\t- Допускаем H0')]
    
    if illustrate:
        print()
        n_first = len(first)
        n_second = len(second)
        mu_first = np.mean(first)
        mu_second = np.mean(second)
        sigma_first = np.std(first)
        sigma_second = np.std(second) 
        SE_first = (sigma_first / np.sqrt(n_first))
        SE_second = (sigma_second / np.sqrt(n_second))
        SE_sub = np.sqrt(((sigma_first ** 2) / n_first) + ((sigma_second ** 2) / n_second))
        range_inside_alpha = st.norm.interval(1 - alpha, 0, 1)
        range_inside_95 = st.norm.interval(0.95, 0, 1)
        alpha_critical = [SE_sub * r for r in range_inside_alpha]
        conf_first = [mu_first + (SE_first * r) for r in range_inside_95]
        conf_second = [mu_second + (SE_second * r) for r in range_inside_95]

        fig = plt.figure(figsize = (18, 4))
        grid = fig.add_gridspec(ncols = 12, nrows = 6)
        ax_norm = fig.add_subplot(grid[:, :7])
        ax_conf = fig.add_subplot(grid[1:5, 7:])
        for ax, title in zip([ax_norm, ax_conf], 
                             ['Распределение разностей выборок при допущении H0', 
                              'Доверительные интервалы выборок (95%)']):
            ax.tick_params(labelleft = False)
            ax.grid(b = False, axis = 'y')
            ax.grid(color = 'grey', linestyle = '-', alpha = 0.2, axis = 'x', zorder = 0)
            ax.set_title(title, fontsize = 20, y = 1.02)

        xmin = -3 * SE_sub
        xmax = 3 * SE_sub
        array_of_dots = np.linspace(xmin, xmax, 100)
        ax_norm.plot(array_of_dots, st.norm.pdf(array_of_dots, 0, SE_sub), linewidth = 5, 
                     color = '#d5ffff', zorder = 1)
        ax_norm.set_ylim(ymin = 0)
        x_ticks = [xmin, -2 * SE_sub, -1 * SE_sub, 0, SE_sub, 2 * SE_sub, xmax]
        ax_norm.set(xlim = (xmin, xmax), xticks = x_ticks, 
                    xticklabels = pd.Series([round(x, x_dec) if (x_dec != 0) else int(round(x, x_dec)) 
                                             for x in x_ticks]).replace(0, '0\n(H0)').to_list())
        ax_conf.set(ylim = (-4, 4), yticks = np.arange(-4, 5), xlabel = col_name)

        ax_norm.vlines(x = alpha_critical, ymin = ax_norm.get_ylim()[0], ymax = ax_norm.get_ylim()[1],
                       linestyle = '--', lw = 2, color = '#ff474c', alpha = 0.4)
        left_ha = 'right'
        right_ha = 'left'
        mult = 1.05
        if alpha < 0.03:
            left_ha = 'left'
            right_ha = 'right'
            mult = 0.95
        ax_norm.text(alpha_critical[0] * mult, ax_norm.get_ylim()[1] * 0.8, 
                     f'alpha\n{round(alpha/2, 3)}\n{round(range_inside_alpha[0], 2)} SE', 
                     ha = left_ha, color = '#ff474c', 
                     va = 'center', alpha = 0.7, fontsize = 15,  fontweight = 'bold')
        ax_norm.text(alpha_critical[1] * mult, ax_norm.get_ylim()[1] * 0.8, 
                     f'alpha\n{round(alpha/2, 3)}\n{round(range_inside_alpha[1], 2)} SE', 
                     ha = right_ha, color = '#ff474c', 
                     va = 'center', alpha = 0.7, fontsize = 15,  fontweight = 'bold')
        left_fill = np.linspace(xmin, alpha_critical[0], 50)
        ax_norm.fill_between(left_fill, st.norm.pdf(left_fill, 0, SE_sub), 0, color = '#ff474c', alpha = 0.7)
        right_fill = np.linspace(alpha_critical[1], xmax, 50)
        ax_norm.fill_between(right_fill, st.norm.pdf(right_fill, 0, SE_sub), 0, color = '#ff474c', alpha = 0.7)
        sub = max(mu_first, mu_second) - min(mu_first, mu_second)
        shift = ax_norm.get_ylim()[1] * 0.03
        if sub > alpha_critical[1]:
            color = 'red'
            if sub > xmax:
                sub = xmax
                y = [shift * 2.5, shift * 2.5]
                mk = 'x'
                s = 400
            else:
                y = [0, 0]
                mk = '|'
                s = 1200
        else:
            color = '#d5ffff'
            y = [0, 0]
            mk = '|'
            s = 1200
        ax_norm.scatter(y = y, x = [sub, sub * -1], marker = mk, color = color, 
                        s = s, linewidth = 5, alpha = 0.85, zorder = 2)
        ax_norm.annotate(text = 'разности\nданных\nвыборочных\nсредних', xy = (sub * 0.95, y[0] + shift),   
                         xytext = (0, ax_norm.get_ylim()[1] * 0.5), ha = 'center', color = '#d5ffff', alpha = 0.7, 
                         fontsize = 15, arrowprops = {'width': 3, 'color': '#d5ffff', 'alpha': 0.3})
        ax_norm.annotate(text = 'разности\nданных\nвыборочных\nсредних', xy = (sub * -0.95, y[0] + shift), 
                         xytext = (0, ax_norm.get_ylim()[1] * 0.5), ha = 'center', color = '#d5ffff', alpha = 0.7, 
                         fontsize = 15, arrowprops = {'width': 3, 'color': '#d5ffff', 'alpha': 0.3})

        for y, mu, conf, color, name in zip([2, 0], [mu_first, mu_second], [conf_first, conf_second], 
                                            ['#2976bb', '#7ebd01'], names):
            ax_conf.hlines(y = y, xmin = conf[0], xmax = conf[1], color = color, lw = 3, zorder = 1)
            ax_conf.vlines(x = conf, ymin = y - 0.4, ymax = y + 0.4, color = color, lw = 1.5, zorder = 1)
            ax_conf.scatter(mu, y, color = '#d5ffff', s = 100, linewidths = 2, edgecolors = color, 
                            label = name, zorder = 2)
        ax_conf.legend(loc = 'lower center', fontsize = 'xx-large', ncol = 2)
        ax_conf.hlines(y = [0, 2], xmin = ax_conf.get_xlim()[0], xmax = ax_conf.get_xlim()[1], 
                       linestyle = 'dotted', alpha = 0.5, zorder = 0, color = 'white', lw = 2)

        display(fig)
        plt.close(fig)
#%%
def qq_plot(samp, no_qq_table = True, alpha = None, title = False, ax = False, real_SE = None):
    """
    Строит qq-plot, проводит три теста на нормальность.
    Если не указать 'qq_table = False', выводит первые и последние 
    четыре строчки таблицы, переменные которой отображены на qq-plot. Первый позиционный 
    аргумент - значения выборки (одномерная коллекция).
    """
    if 'pd' not in dir():
        import pandas as pd
    if 'np' not in dir():
        import numpy as np
    if 'st' not in dir():
        import scipy.stats as st
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'display' not in dir():
        from IPython.display import display
    from warnings import filterwarnings
    filterwarnings("ignore")
    
    """
    Для xlim и ylim при наличии выбросов (наработка)
    
    if maximal > max(xticks.keys()):
        lim_max = max(xticks.keys())
        ax_qq.set_ylim(ymax = lim_max)
        ax_hist.set_xlim(xmax = lim_max)
        over_count = (samp > lim_max).sum()
        ax_qq.text(ax_qq.get_xlim()[1] * 0.98, lim_max * 0.4, f'+{over_count} точек,\nпревышающих 10 std', 
                   ha = 'right', va = 'top', c = '#95d0fc', fontsize = 14)
        ax_hist.text(lim_max, ax_hist.get_ylim()[1] * 0.6, f'+{over_count} значений выше 10 std\nза пределами графика', 
                   ha = 'right', va = 'center', c = '#95d0fc', fontsize = 14)
        
    if minimal > min(xticks.keys()):
        lim_min = min(xticks.keys())
        ax_qq.set_ylim(ymin = lim_min)
        ax_hist.set_xlim(xmin = lim_min)
        over_count = (samp > lim_min).sum()
        ax_qq.text(ax_qq.get_xlim()[1] * 0.98, lim_max * 0.6, f'+{over_count} точек,\nпревышающих 10 std', 
                   ha = 'right', va = 'bottom', c = '#95d0fc', fontsize = 14)
        ax_hist.text(lim_min, ax_hist.get_ylim()[1] * 0.6, f'+{over_count} значений выше 10 std\nза пределами графика', 
                   ha = 'left', va = 'center', c = '#95d0fc', fontsize = 14)
    """

    samp = np.array(samp.sort_values().dropna().reset_index(drop = True))
    leng = len(samp)
    mu = np.mean(samp)
    sigma = np.std(samp)
    
    calc_quantile = lambda x: x / (leng + 1)
    calc_std = lambda x: st.norm.ppf(x, 0, 1)
    calc_exp = lambda x: mu + (x * sigma)
    
    quantile = calc_quantile(np.arange(1, leng + 1))
    exp_std = calc_std(quantile)
    exp_val = calc_exp(exp_std)
    real_std = st.zscore(samp)

    if ax == False:
        fig, ax_qq = plt.subplots(figsize = (7.5, 4.5))
        no_initial_ax = True
    else:
        ax_qq = ax
        no_initial_ax = False
        
    if title:
        ax_qq.set_title(title, y = 1.15, fontsize = 15)
    ax_qq.set(ylabel = 'шкала значений переменной', xlabel = 'ожидание на квантиле (в std)')
    ax_qq.grid(b = True, linewidth = 0.4, color = 'grey', alpha = 0.4)
    ax_qq.scatter(exp_std, samp, s = 25, zorder = 4, label = 'значение выборки', c = '#789b73')
    ax_qq.plot([np.min(exp_std), np.max(exp_std)], [np.min(exp_val), np.max(exp_val)], 
               c = '#ff000d', lw = 3, zorder = 2, label = 'нормальное распределение')
    
    if real_SE:
        theoretical_SE = dict([(mu + (i * sigma), i) for i in real_SE if 
                       ((mu + (i * sigma) < np.max(samp)) and (mu + (i * sigma) > np.min(samp)))])
    else:
        theoretical_SE = dict([(mu + (i * sigma), i) for i in np.arange(-10, 11) if 
                       ((mu + (i * sigma) < np.max(samp)) and (mu + (i * sigma) > np.min(samp)))])

    samp_se_real_values = {}
    for key, val in theoretical_SE.items():
        samp_se_real_values[val] = np.argmin(np.abs(samp - key))
        # argmin возвращает индексы значений, отклонение которыз наиболее близко к 
        # круглой цифре: отклонению в 0 SE, 1 SE, 2 SE и так до десяти.
    
    ax_grid = ax_qq.twiny()
    ax_grid.tick_params(bottom = False, labelbottom = False, top = True, labeltop = True)
    ax_grid.xaxis.set_label_position('top')
    ax_grid.set_xlabel('значение std на квантиле (округлено)')
    ax_grid.grid(b = True, axis = 'x', linewidth = 3, color = '#789b73', alpha = 0.6)
    
    empirical_SE = {}
    for ind in samp_se_real_values.values():
        x_position = exp_std[ind]
        empirical_SE_value = round(real_std[ind], 1)
        empirical_SE.update({x_position: empirical_SE_value})
    
    ax_grid.set_xlim(ax_qq.get_xlim())
    ax_grid.set_xticks(list(empirical_SE.keys()))
    ax_grid.set_xticklabels([int(a) if (a % 1) == 0 else a for a in 
                             np.array(list(empirical_SE.values()))])
    ax_qq.legend(fontsize = 'medium')    
    
    if no_initial_ax:     
        display(fig)
        plt.close(fig)
    
    if not no_qq_table:
        display(pd.concat([pd.Series(quantile[:4]), pd.Series(exp_std[:4]), pd.Series(exp_val[:4]), 
                           pd.Series(samp[:4]), pd.Series(real_std[:4])], axis = 1)
                .set_axis(['Квантиль', 'Ожидание на квантиле (в std)', 
                           'Ожидание на квантиле (по мю и сигме)', 'Значение выборки на квантиле',
                           'Значение std на квантиле'], axis = 1))
        display(pd.concat([pd.Series(quantile[-4:]), pd.Series(exp_std[-4:]), pd.Series(exp_val[-4:]), 
                           pd.Series(samp[-4:]), pd.Series(real_std[-4:])], axis = 1)
                .set_axis(['Квантиль', 'Ожидание на квантиле (в std)', 
                           'Ожидание на квантиле (по мю и сигме)', 'Значение выборки на квантиле',
                           'Значение std на квантиле'], axis = 1)
                .set_axis([-4, -3, -2, -1], axis = 0))
    
    if alpha:
        shapiro = st.shapiro(samp)[1]
        print('{: >41}'.format('pvalue'))
        print('{: <35}'.format('Тест Шапиро - Уилка:'), shapiro)
        ks = st.kstest(real_std, 'norm')[1]
        print('{: <35}'.format('Тест Колмогорова - Смирнова:'), ks)
        if leng > 2000:
            jb = st.jarque_bera(samp)[1]
            print('{: <35}'.format('Тест Харке - Бера:'), jb)
        else:
            jb = 0

        if all([x > alpha for x in [ks, shapiro, jb]]):
            print('\t- Не получилось отвергнуть нулевую гипотезу, переменная распределена нормально')
        elif all([x < alpha for x in [ks, shapiro, jb]]):
            print('\t- Отвергаем нулевую гипотезу: распределение не нормально')
        else:
            print('\t- Отличия от нормальности неоднозначны')
#%%
def page(URL, headers = {}, open_in_chrome = False):
    """
    Первый аргумент - url, параметр headers= принимает словарь,
    который передаётся одноимённому параметру функции requests.get().
    Если передать open_in_chrome = True, то в отдельном окне браузера
    откроется страница, созданная на основе полученного html-кода.
    Если ответ в формате html, то функция вернёт объект bs4, если в 
    json - то словарь. Если в ином формате - то объект Response.
    """
    if 'req' not in dir():
        import requests as req
    if 'BS' not in dir():
        from bs4 import BeautifulSoup as BS
    response = req.get(URL, headers = headers)
    content_type = response.headers['Content-Type']
    if response.ok:
        if 'html' in content_type:
            page = BS(response.text, 'lxml')
            html = True
        elif 'json' in content_type:
            page = response.json()
            html = False
        else:
            print('Ответ от сервера в неизвестном формате: ', content_type)
            return response
    else:
        print('Status code: ', response.status_code)
        return
    
    if open_in_chrome and html:
        if 'Chrome' not in dir():
            from selenium.webdriver import Chrome
        driver_path = 'C:\\Xozyain\\Documents\\Directory\\Python_R\\parsing\\chromedriver.exe'
        page_path = 'C:\\Xozyain\\Documents\\Directory\\Python_R\\parsing\\temp.html' 
        with open(page_path, 'w', encoding = 'utf8', newline = '') as temp:
            temp.write(response.text)
        Chrome(driver_path).get(page_path)
    elif open_in_chrome:
        print('Ответ в формате json, в браузере открывать нечего')
        
    return page
#%%
def linear_regression(col1, col2, baloons = True):


    col1 = np.array(col1.sort_values().reset_index(drop = True))
    col2 = np.array(col2.sort_values().reset_index(drop = True))
    
    if len(col1) != len(col2):
        print('Выровняй коллекции по размерам')
        if len(col1[np.isnan(col1)]) != 0 or len(col2[np.isnan(col2)]) != 0:
            print('Разберись с пропусками')
        return
    if len(col1[np.isnan(col1)] != 0) or len(col2[np.isnan(col2)]):
        print('Разберись с пропусками')
        return
    
    mean_x = np.mean(col1)
    mean_y = np.mean(col2)
    
    r = st.pearsonr(col1, col2)
    r2 = r ** 2
    
    fig = plt.figure(figsize = (18, 12))
    grid = fig.add_gridspec(ncols = 3, nrows = 6, hspace = 0)
    ax_model = fig.add_subplot(grid[:3, :6])
    ax_scatter = fig.add_subplot(grid[4:, 0])
    ax_hist = fig.add_subplot(grid[4:, 1])
    ax_qq = fig.add_subplot(grid[4:, 2])
    
    ax_scatter.set_title('Скедастичность', y = 1.1, fontsize = 18)
    ax_hist.set_title('Распределение остатков', y = 1.1, x = 1.08, fontsize = 18)


def regression_formula(col1, col2, mean_x, mean_y, r):
    b1 = r * (np.std(col2) / np.std(col1))
    b0 = mean_y - (b1 * mean_x)
    predict_y = lambda x: b0 + (b1 * x)
    exp_y = predict_y(col1)
    residuals = col2 - exp_y
    # Возвращает координаты у всех точек регрессионной прямой и остатки всех точек  
    return exp_y, residuals

def draw_model(col1, col2, exp_y, ax, baloons, mean_x, mean_y, r, r2):
    sizes = pd.concat([col1, col2], axis = 1)
    sizes.columns = ['x', 'y']
    ax.set_title(' '.join(['Регрессионная модель\n$r_{xy} =$', 
                            f'{round(r, 4)},', 
                            '$r^2 = $', 
                            f'{round(r2, 4)}']), fontsize = 20)
    if len(col1) > 250:
        try: 
            # Через append добавляю средние значения по х и у, чтобы получить координаты для отрисовки линий
            # Через блок try - потому что может выползти ошибка, если значения не будут делиться на заданное
            # число корзин
            sizes['on_x'] = pd.cut(sizes.iloc[:, 0].append(pd.Series(mean_x)), 
                                   80, labels = [i for i in range(1, 81)])
            sizes['on_y'] = pd.cut(sizes.iloc[:, 1].append(pd.Series(mean_y)), 
                                   45, labels = [i for i in range(1, 46)])
            mod_mean_x = sizes['on_x'].iloc[-1]
            mod_mean_y = sizes['on_y'].iloc[-1]
            sizes = sizes.iloc[:-1, :]
            cut_made = True
        except:
            cut_made = False
    else:
        cut_made = False
    
    if cut_made and baloons:
        total_rows = sizes.shape[0]
        sizes = sizes.groupby(['on_x', 'on_y']).size().reset_index(name = 'counts')
        sizes = sizes[ sizes['counts'] > 0 ]
        sizes['prcs'] = sizes['counts'].div(total_rows)
        ax.set_xticks([i for i in range(1, 81)])
        ax.set_yticks([i for i in range(1, 46)])
        
        ax.set_xticklabels
        ax.set_yticklabels
        
        ax.scatter(x = sizes['on_x'], y = sizes['on_y'], s = sizes['prcs'].mul(200000), alpha = 0.4)
        ax_model.vline
        ax_model.hline
    else:
        ax.scatter(x = col1, y = col2, alpha = 0.5)
        ax_model.vline
        ax_model.hline

        
    def draw_qq():    
        calc_quantile = lambda x: x / (leng + 1)
        calc_std = lambda x: st.norm.ppf(x, 0, 1)
        calc_exp = lambda x: mu + (x * sigma)
        
        quantile = calc_quantile(np.arange(1, leng + 1))
        exp_std = calc_std(quantile)
        exp_val = calc_exp(exp_std)
        real_std = st.zscore(samp)
    
    
        ax_qq.set(ylabel = 'шкала значений переменной', xlabel = 'ожидание на квантиле (в std)')
        ax_qq.grid(linewidth = '0.5', color = 'grey', alpha = 0.4)
        ax_qq.scatter(exp_std, samp, zorder = 4, label = 'значение выборки', c = '#95d0fc')
        ax_qq.plot([np.min(exp_std), np.max(exp_std)], [np.min(exp_val), np.max(exp_val)], 
                   c = 'red', lw = 3, zorder = 2, label = 'нормальное распределение')
        
        xticks = dict([(mu + (i * sigma), i) for i in np.arange(-10, 11) if ((mu + (i * sigma) < np.max(samp)) and (mu + (i * sigma) > np.min(samp)))])
            
        samp_se_real_values = {}
        for key, val in xticks.items():
            samp_se_real_values[val] = np.argmin(np.abs(samp - key))
    
        half = np.median(list(samp_se_real_values.keys()))
        shift = ax_qq.get_ylim()
        shift = (shift[1] - shift[0]) * 0.1
        for ind in samp_se_real_values.values():
            x = exp_std[ind]
            y = samp[ind]
            text = real_std[ind]
            text = [round(text, 1) if round(text, 1) != int(text) else int(text)][0]
            ax_qq.scatter(x, y, c = '#95d0fc', s = 600, lw = 3, marker = '|', zorder = 3, alpha = 0.5)
            if text < half:
                ax_qq.text(x, y + shift, text, fontsize = 15, ha = 'center', c = '#95d0fc' )
            else:
                ax_qq.text(x, y - shift, text, fontsize = 15, ha = 'center', c = '#95d0fc' )
        
        ax_qq.scatter([], [], c = '#95d0fc', s = 600, lw = 3, marker = '|', zorder = 3, alpha = 0.5,
                     label = 'действительное\nстандартное отклонение')
        ax_qq.legend(fontsize = 'x-large')    
        
        drawn = ax_hist.hist(samp, bins = bins, color = '#95d0fc', alpha = 0.8, density = True,
                             label = 'гистограмма выборки', zorder = 2)
        ax_hist.set(xlabel = 'значения выборки', xticks = [round(x, 2) for x in list(xticks.keys())], 
                   yticks = ax_hist.get_yticks()[1:])
        minimal = np.min(drawn[1])
        maximal = np.max(drawn[1])
        array_of_dots = np.linspace(minimal, maximal, bins)
        
        ax_double = ax_hist.twiny()
        ax_double.set(xlim = ax_hist.get_xlim(), xticks = list(xticks.keys()), xticklabels = list(xticks.values()))
        
        ax_hist.plot(array_of_dots, st.norm.pdf(array_of_dots, mu, sigma), lw = 3, color = 'red', 
                zorder = 3)
        ax_hist.legend(fontsize = 'x-large')
            
        ax_double.tick_params(labelleft = False)
        ax_hist.tick_params(labelleft = False, labelright = True)
        ax_hist.grid(b = False, axis = 'x')
        ax_double.grid(b = False, axis = 'y')
        ax_double.grid(axis = 'x', zorder = 4, color = 'grey', lw = 2.5, alpha = 0.25)
    
    
    display(fig)
    plt.close(fig)
#%%
def my_plotly_config(zoom = False, display_bar = True, updates = None):
    my_plotly_config = {'displaylogo': False,
                        'scrollZoom': zoom,
                        'displayModeBar': display_bar,
                        'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'hoverCompareCartesian'
                                                   'hoverClosestCartesian', 'autoScale2d', 'toggleSpikelines'],
                        'toImageButtonOptions': {'height': None, 'width': None}}
    if updates:
        my_plotly_config.update(updates)
    return my_plotly_config
#%%
def two_subplots(figsize, suptitle, titles):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = figsize)
    fig.suptitle(suptitle, fontsize = 18, y = 1.15)
    grid = fig.add_gridspec(1, 2, wspace = 0.25)
    ax1 = fig.add_subplot(grid[0, 0]); ax2 = fig.add_subplot(grid[0, 1])
    for ax, title in zip([ax1, ax2], titles):
        ax.set_title(title, y = 1.05, fontsize = 15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig, ax1, ax2
#%%
def my_display(df, index_name = None, col_names = None):
    current_index = df.index.name
    if current_index:
        df = df.reset_index().rename({current_index: index_name}, axis = 1).set_index(index_name)
    else:
        df = df.reset_index().rename({'index': index_name}, axis = 1).set_index(index_name)
    if col_names:
        df.columns = col_names
    display(df)
#%%
def ab_conversion(daily, date_col, group_col, clients_cum_col, conv_col, mapper, figsize, return_fig = False, alpha = 0.05, legend = ['A', 'B']):
    import scipy.stats as st
    import numpy as np 
    import matplotlib.pyplot as plt
    
    def two_subplots(figsize, titles):
        fig = plt.figure(figsize = figsize)
        grid = fig.add_gridspec(1, 2, wspace = 0.25)
        ax1 = fig.add_subplot(grid[0, 0]); ax2 = fig.add_subplot(grid[0, 1])
        for ax, title in zip([ax1, ax2], titles):
            ax.set_title(title, y = 1.05, fontsize = 15)
        return fig, ax1, ax2
    
    def customize_ax(ax, ylabel, xticks, figsize):
        ax.set(xlim = (0, np.max(xticks) + 1), xticks = xticks, ylabel = ylabel, 
               xlabel = 'день эксперимента', ylim = ax.get_ylim())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if figsize[0] > 13:
            fontsize = 'large'
        else:
            fontsize = 'medium'            
        ax.legend(fontsize = fontsize)
        ax.grid(b = True, axis = 'y')
    
    fig, ax1, ax2 = two_subplots(figsize, ['накопительная C1', 'разница между группами'])
    xticks = np.arange(1, daily[date_col].drop_duplicates().shape[0] + 1)

    for group, color, legend in zip(['A', 'B'], ['#9a0200', '#789b73'], legend):
        data_sliced = daily[ daily[group_col] == mapper[group] ]
        ax1.plot(xticks, data_sliced[conv_col], label = f'группа {legend}', 
                 color = color, linewidth = 2.5)
    customize_ax(ax1, 'накопительная C1, %', xticks, figsize)
    ax1.vlines(xticks, ax1.get_ylim()[0], data_sliced[conv_col], alpha = 0.5, linestyle = 'dotted')

    difference_overall = (daily.loc[ daily[group_col] == mapper['B'], conv_col ].values 
                          - daily.loc[ daily[group_col] == mapper['A'], conv_col ].values)
    ax2.plot(xticks, difference_overall, linewidth = 3, color = 'grey')
    ax2.axhline(y = 0, color = 'k', linestyle = '--')
    at_least_one_h1_005 = False
    at_least_one_h1_001 = False
    
    for xtick, date in enumerate(daily[date_col].drop_duplicates(), 1):
        samp_date = daily[ (daily[date_col] == date) ]
        a_buy = samp_date.loc[ (samp_date[group_col] == mapper['A']) ][clients_cum_col].values[0]
        b_buy = samp_date.loc[ (samp_date[group_col] == mapper['B']) ][clients_cum_col].values[0]
        a_vis = a_buy / (samp_date.loc[ (samp_date[group_col] == mapper['A']) ][conv_col].values[0])
        b_vis = b_buy / (samp_date.loc[ (samp_date[group_col] == mapper['B']) ][conv_col].values[0])
        P = (a_buy + b_buy) / (a_vis + b_vis)
        P1 = a_buy / a_vis
        P2 = b_buy / b_vis
        difference = P2 - P1
        scale = np.sqrt(((1/a_vis) + (1/b_vis)) * P * (1 - P))
        conf = st.norm.interval(1 - alpha, loc = difference, scale = scale)
        z_score =  difference / scale
        p_value = (1 - st.norm(0, 1).cdf(abs(z_score))) * 2
        dot = False
        if p_value < alpha:
            style = {'color': 'green', 'alpha': 0.8, 'linewidth': 2}
            at_least_one_h1_005 = True
            if p_value < 0.01:
                dot = True
                if alpha > 0.01:
                    at_least_one_h1_001 = True
        else:
            style = {'color': 'grey', 'alpha': 0.6}
        ax2.vlines(xtick, conf[0], conf[1], **style)
        if dot:
            ax2.scatter(xtick, difference, color = 'green', s = 50)

    inscr = round((1 - alpha) * 100, 2)
    inscr = [inscr if (inscr % 1 != 0) else int(inscr)][0]
    ax2.vlines([], [], [], color = 'grey', alpha = 0.6, label = f'{inscr}%-ый доверительный\nинтервал разницы долей')
    if at_least_one_h1_005:
        ax2.vlines([], [], [], color = 'green', alpha = 0.8, linewidth = 2, label = f'p-value < {round(alpha, 3)}')
        if at_least_one_h1_001:
            ax2.scatter([], [], color = 'green', s = 50, label = 'p-value < 0.01')
    customize_ax(ax2, 'разница долей, %', xticks, figsize)
    
    if return_fig:
        return fig
    display(fig)
    plt.close(fig)
#%%
def detail_tag_in_portfolio(st):
    current_line = '<details><summary><strong>Открыть выводы</strong></summary><ol style="padding-left: 0px;">'
    global_postfix = '</ol></details>'
    for line in st:
        prefix = '<li><p align="justify">'
        postfix = '</p></li>'
        current_line = ''.join([current_line, prefix, line, postfix])
    current_line = ''.join([current_line, global_postfix])
    return current_line

"""
a = ""
b = ""
c = ""
d = ""
e = ""
f = ""
g = ""
h = ""
i = ""
j = ""
k = ""
l = ""
m = ""
n = ""
o = ""

detail_tag_in_portfolio([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o])
"""
#%%
def my_info(df):
    import pandas
    import numpy
    info = pd.DataFrame(index = df.columns, columns = ['type', 'not_null_cnt', 'nunique', 'mean', 'min', 
                                                       '25%', '50%', '75%', 'max'])
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        info.loc[col, 'type'] = df[col].dtype
        info.loc[col, 'not_null_cnt'] = df[col].notna().sum()
        if col in numeric_cols:
            nuniq = df[col].nunique()
            if nuniq / df[col].shape[0] > 0.1:
                info.loc[col, 'nunique'] = '-'
            else:
                info.loc[col, 'nunique'] = nuniq
            info.loc[col, 'mean'] = df[col].mean()
            info.loc[col, 'min'] = df[col].min()
            info.loc[col, 'max'] = df[col].max()
            info.loc[col, '25%'] = df[col].quantile(0.25)
            info.loc[col, '50%'] = df[col].quantile(0.5)
            info.loc[col, '75%'] = df[col].quantile(0.75)
        else:
            info.loc[col, 'nunique'] = df[col].nunique()
            info.iloc[:, 3:] = '-'
    return info
#%%
def corr_matrix(df, half_table = False):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr_matrix = df.corr().round(2)
    if half_table:
        corr_matrix_mask = corr_matrix.copy()
        corr_matrix_mask.iloc[:, :] = False
        corr_matrix_mask = corr_matrix_mask.values
        np.fill_diagonal(corr_matrix_mask, True)
        for i, row in enumerate(corr_matrix_mask):
            diag_cell_found = False
            for j, val in enumerate(row):
                if diag_cell_found:
                    corr_matrix_mask[i, j] = True
                elif val == True:
                    diag_cell_found = True
                    corr_matrix_mask[i, j] = False
    else:
        corr_matrix_mask = None
    fig, ax = plt.subplots(figsize = ([0.8 * corr_matrix.shape[1] if corr_matrix.shape[1] >= 9 else 7][0], 
                                      [0.4 * corr_matrix.shape[1] if corr_matrix.shape[1] >= 9 else 2.5][0]))
    sns.heatmap(corr_matrix.abs(), annot = corr_matrix, mask = corr_matrix_mask, cmap = 'Blues', cbar = False, ax = ax)
    ax.set_title('Корреляция между числовыми столбцами', y = [1.05 if half_table else 1.1][0], fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 50, ha = 'right')
    display(fig)
    plt.close(fig)
#%%
def draw_roc(y_test, probabilities, mode_05 = '', title = ''):
    """
    По умолчанию рисует ROC-кривую и отмечает порог, наиболее близкий к 0.5 точкой. 
    Варианты mode_05: 'explain', 'point', 'hide'.
    """
    import pandas as pd
    import numpy as np
    import plotly
    from plotly import graph_objs as go
    from sklearn.metrics import recall_score, roc_auc_score, roc_curve, confusion_matrix
    from IPython.display import display 
    
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    text = []
    for thr in thresholds:
        y_pred = (probabilities >= thr).astype('int')
        ((tn, fp), (fn, tp)) = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        thr_str = '0.0...' if round(thr, 4) == 0 else round(thr, 4) 
        text_thr = f'<b>Порог: {thr_str}</b><br><br>TN: {tn}, FP: {fp}<br>FN: {fn}, TP: {tp}'
        text.append(text_thr)
    index_05 = abs(0.5 - thresholds).argmin()

    def return_layout(mode_05, roc_auc):
        layout = dict(xaxis = dict(range = (-0.025, 1)), yaxis = dict(range = (0, 1.05)), template = 'ggplot2',
                  showlegend = False, height = 450, width = 550, title = dict(text=f"<b>{title}</b>", y = 1,
                  xanchor = 'center', x = 0.5, yref = 'paper', yanchor = 'bottom', font_size = 16, pad = {'b': 20}, 
                  xref = 'paper'), xaxis_title = dict(text ='fpr (1 - специфичность)', font = dict(size = 14.5)), 
                  margin = dict(l = 0, r = 10, t = 70, b = 0), yaxis_title = dict(text ='tpr (полнота)',
                  font = dict(size = 14.5)))
        annotations = [dict(text = f'ROC AUC: {round(roc_auc, 3)}', font_color = 'black', opacity = 0.5,
                  x = 0.8, y = 0.1, showarrow = False, xref = 'x', yref = 'y', font_size = 12)]
        if mode_05 == 'explain':
            annot = ('<b>Дефолтный порог 0.5</b><br>(когда предсказанная вероятность единицы '
                     'больше<br>или равна 0.5, классифицировать как единицу)')
        elif mode_05 == 'point':
            annot = '<b>Дефолтный порог 0.5</b>'
        else:
            layout.update(dict(annotations = annotations))
            return layout
        annotations.append(dict(text = annot,  x = fpr[index_05], y = tpr[index_05],
                  ax = fpr[index_05] + 0.4, ay = tpr[index_05] - 0.2, bgcolor = 'rgba(170, 170, 170, 0.5)', 
                  arrowhead = 6, showarrow = True, axref = 'x', ayref = 'y', arrowwidth = 3, font_size = 12, 
                  borderpad = 10))
        layout.update(dict(annotations = annotations))
        return layout

    data = [go.Scatter(x = fpr, y = tpr, text = text, mode = 'lines', fill = 'tozeroy', marker_color = 'green', 
                       fillcolor = 'rgba(0, 150, 0, 0.25)', hoverlabel = {'bgcolor': 'rgba(170, 170, 170, 0.3)'},
                       hovertemplate = '<extra></extra>%{text}<br>TPR: %{y:.3f}<br>FPR: %{x:.3f}')]
    if mode_05 != 'hide':
        data.append(go.Scatter(x = [fpr[index_05]], y = [tpr[index_05]], mode = 'markers', 
                               hoverlabel = {'bgcolor': 'rgba(170, 170, 170, 0.3)'},
                               marker = dict(color = 'green', size = 10), text = [text[index_05]],
                               hovertemplate = '<extra></extra>%{text}<br>TPR: %{y:.3f}<br>FPR: %{x:.3f}'))

    fig = go.Figure(data = data, layout = return_layout(mode_05, roc_auc))
    fig.show(config = {'displaylogo': False, 'scrollZoom': False, 'displayModeBar': True,
                        'modeBarButtonsToRemove': ['zoom2d', 'lasso2d', 'hoverCompareCartesian'
                                                   'hoverClosestCartesian', 'autoScale2d', 'toggleSpikelines'],
                        'toImageButtonOptions': {'height': None, 'width': None}})
#%%
def features_importance(model, model_type, title, X, y, 
                        ax = False, return_perm = False, scoring = 'r2', random_state = 0):
    import pandas as pd
    import numpy as np
    import matplotlib.patheffects as path_effects
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    from IPython.display import display
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    
    color_metrica = '#edffec'
    color_train = '#1f441e'
    color_test = '#bd2000'
    bot_label = f'permutation importance ({scoring})'

    if model_type == 'linear':
        top_label = 'коэффициенты по модулю'
        metrica = model.coef_[0]
    elif model_type == 'forest':
        top_label = 'MDI'
        metrica = model.feature_importances_

    result_train_sample = permutation_importance(model, X_train, y_train, scoring = scoring)
    result_test_sample = permutation_importance(model, X_test, y_test, scoring = scoring)
    impact = pd.DataFrame(dict(zip(X_test.columns, metrica)), 
                          index = ['metrica']).T.sort_values(by = 'metrica', ascending = False)
    impact = impact.reset_index().rename({'index': 'col'}, axis = 1)
    if model_type == 'linear':
        impact = impact.loc[(impact.metrica.abs().sort_values(ascending = False).index), :].reset_index(drop = True)
    impact['perm_train'] = impact['col'].map(dict(zip(X_test.columns, result_train_sample.importances_mean)))
    impact['std_train'] = impact['col'].map(dict(zip(X_test.columns, result_train_sample.importances_std)))
    impact['perm_test'] = impact['col'].map(dict(zip(X_test.columns, result_test_sample.importances_mean)))
    impact['std_test'] = impact['col'].map(dict(zip(X_test.columns, result_test_sample.importances_std)))
    impact['train_xerr_min'] = impact['perm_train'].sub(impact['std_train'])
    impact['train_xerr_max'] = impact['perm_train'].add(impact['std_train'])
    impact['test_xerr_min'] = impact['perm_test'].sub(impact['std_test'])
    impact['test_xerr_max'] = impact['perm_test'].add(impact['std_test'])

    if model_type == 'linear':
        color_metrica = ['#edffec' if x >= 0 else '#f5c0c0' for x in impact.metrica.values]
        impact['metrica'] = impact.metrica.abs()
        
    if ax == False:
        fig, ax = plt.subplots(figsize = (9, 0.45 * impact.shape[0]))
        no_initial_ax = True
    else:
        no_initial_ax = False

    xticks = np.linspace(0, round(impact.metrica.max(), 2), 6).round(3)
    ax.set(ylim = (impact.index.max() + 1, -1), yticks = impact.index, xticks = xticks, xlabel = top_label)
    ax.set_yticklabels(impact.col, fontsize = 12.5)
    ax.set_xticklabels(xticks, color = '#edffec', weight = 'bold', fontsize = 11.5, family = 'sans-serif',
                       path_effects = [path_effects.Stroke(linewidth = 2.2, foreground = 'k'), path_effects.Normal()])
    ax.set_title(title, y = 1.12, fontsize = 16)

    ax_perm = ax.twiny()
    ax.tick_params(bottom = False, labelbottom = False, top = True, labeltop = True)
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = 10
    ax_perm.grid(b = True, axis = 'x', linewidth = 3, color = 'grey', alpha = 0.4)
    ax_perm.tick_params(bottom = True, labelbottom = True, top = False, labeltop = False)
    ax_perm.xaxis.set_label_position('bottom')
    ax_perm.set(xlim = (-0.18, 0.6), xticks = np.arange(0, 0.61, 0.1))
    ax_perm.set_xlabel(bot_label, x = 0.63, color = color_test)
    ax_perm.set_xticklabels(ax_perm.get_xticks().round(1), color = color_test, weight = 'bold', fontsize = 12.5, 
                            family = 'sans-serif')
    
    if model_type == 'linear':
        ax_perm.barh([], [], label = 'Положительный коэффициент')
        ax_perm.barh([], [], label = 'Отрицательный коэффициент')
    elif model_type == 'forest':
        ax_perm.barh([], [], label = 'MDI признака')
    ax.barh(impact.index, impact.metrica, color = color_metrica, ec = 'k', height = 1, linewidth = 1.5)
    ax_perm.fill_between(x = [0, 0.6], y1 = ax.get_ylim()[0], y2 = ax.get_ylim()[1], color = 'grey', alpha = 0.2)
    ax_perm.scatter(impact.perm_train, np.array(impact.index) - 0.2, c = color_train, s = 60, 
                    label = 'perm.imp. на обучающей выборке')
    ax_perm.hlines(y = np.array(impact.index) - 0.2, xmin = impact.train_xerr_min, 
                   xmax = impact.train_xerr_max, color = color_train, linewidth = 2)
    ax_perm.scatter(impact.perm_test, np.array(impact.index) + 0.2, c = color_test, s = 60,
                   label = 'perm.imp. на валидационной выборке')
    ax_perm.hlines(y = np.array(impact.index) + 0.2, xmin = impact.test_xerr_min, 
                   xmax = impact.test_xerr_max, color = color_test, linewidth = 2, 
                   label = 'std perm.imp. на пяти повторениях')
    ax_perm.hlines(y = np.array(impact.index), xmin = 0, xmax = impact[['perm_train', 'perm_test']].max(axis = 1), 
                   color = 'grey', linewidth = 1)
    
    fontsize = 'medium' if (impact.shape[1] < 16) else 'large'
    leg = ax_perm.legend(fontsize = fontsize, facecolor = '#d8dcd6', loc = 'lower right')
    LH = leg.legendHandles
    if model_type == 'linear':
        LH[-2].set_color('#edffec')
        LH[-1].set_color('#f5c0c0') 
    elif model_type == 'forest':
        LH[-1].set_color('#edffec')
        
    if no_initial_ax:
        display(fig)
        plt.close(fig)
    if return_perm:
        return impact
#%%
def corr_clusters(X, threshold, draw_dendrogram = True, title = ''):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    import pandas as pd
    
    corr = X.corr()
    corr_linkage = linkage(corr, method = 'ward')
    clusters = pd.DataFrame(dict(zip(X.columns, fcluster(corr_linkage, threshold, criterion = 'distance'))), 
                                     index = ['cluster']).T.sort_values(by = 'cluster')
    if draw_dendrogram:
        fig, ax = plt.subplots(figsize = (7, 0.225 * corr.shape[0]))
        dg = dendrogram(corr_linkage, labels = X.columns, leaf_rotation = 90, ax = ax, 
                        color_threshold = threshold, above_threshold_color = 'black')
        ax.set_title(title, fontsize = 16, y = 1.05)
        ax.set(ylabel = 'расстояние по Варду')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        display(fig)
        plt.close(fig)
        
    return clusters
#%%
def remove_correlated_features(model, model_type, X, thr_max):
    actual_features = set(X.columns)
    for thr in np.arange(0, thr_max, 0.1):
        clusters = corr_clusters(X, thr, draw_dendrogram = False)
        if model_type == 'linear':
            clusters['coeffs'] = clusters.index.map(dict(zip(X.columns, np.abs(model.coef_[0]))))
        elif model_type == 'forest':
            clusters['coeffs'] = clusters.index.map(dict(zip(X.columns, model.feature_importances_)))
        new_cols = set(clusters[ clusters.coeffs.isin(clusters.groupby('cluster')['coeffs'].max()) ].index)
        removed_cols = actual_features.difference(new_cols)
        if len(removed_cols) == 0 and thr != 0:
            continue
        actual_features = new_cols
        X_exp = X.loc[:, actual_features]
        metrics = learn(X_exp, y, models = {0: LogisticRegression(random_state = 0)}, return_metrics = True)
        print('{: <45}'.format(', '.join(list(removed_cols)) + [':' if thr != 0 else '<изначальный вариант>'][0]), 
              end = '')
        print(metrics['Матрица ошибок'], '. ROC_AUC: ', round(metrics['ROC_AUC'], 4), sep = '')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

# ФИШКИ СПАЙДЕРА, КОНСОЛИ И КОМАНДНОЙ СТРОКИ
"""
%reset -f - начало нового сеанса
%reset -s -f - очистка пространства имён
%debug - режим отладки
%quickref - короткая справка по магическим командам
%magic - подробная справка по магическим командам
%run ... - выполнить код из файла
%time ... - единоразовое время выполнения (1 млн ns = 1 ms = 1000 µs)
%timeit ... - усреднённое время выполнения
%xdel ... - удалить переменную и ссылки на неё

Контрл+N - новый файл

dir() прописанное в консоли возвращает "выученные"

В консоли: cd.. - папка назад cd Directory - найти по
текущему пути папку Directory и зайти в неё. cd С:\\... -
зайти по адресу

help() в консоли возвращает описание "выученной" команды,
описание возможных действий с "выученной" переменной

стрелочки вверх\вниз в консоли перелистывают недавний
ввод в консоль

проигрывая фрагмент программы в консоли, можно записать
в неё изменённую переменную или функцию

f9 проигрывает выделенную строку, если ничего не
выделено

нажатие на Tab, если начал вводить имя переменной или
функции в консоль, заполнит имя до конца или предложит
выбор и заполнит имя когда выбор будет сделан или его
не останется

выделить фрагмент + Таб - добавляет табуляцию всему
фрагменту

Контрл + Бэкспейс/Делит - удалить всё слово целиком

Контрл + Альт + Стрелка вверх - дублировать строку

Контрл + Д - удалить строку

Шифт + Таб - убрать табуляцию у строки/выделенных строк

Контрл + стрелка влево/вправо - к предыдущему или
следующему слову

Контрл + А(лат) - выделить всё

ячейка обособляется #%% в начале и в конце, текущая ячейка
запускается по команде Контрл + Ентер

Альт + вверх\вниз перемещает выбранную строчку или
выбранный фрагмент кода

Шифт + Контрл + Альт + M(лат) увеличивают до максимального
размера выбранное окно и уменьшает обратно

Контрл + S при курсоре в файле сохраняет его, в консоли -
сохраняет текущую сессию в html-файл

В окне с записанными переменными можно менять их
значение для консоли двойным нажатием

Контрл + Шифт + Таб - переключение на следующий файл
Контрл + Таб - на предыдущий

чтобы проинспектировать выполнение кода в определённом
месте, можно поставить справа от номера строки веху,
запустить дебаггер и переместиться к вехе. в режиме
дебаггера также можно сразу перемещаться к ближайшей
функции или методу

при работе с чанксайзами записывать в csv можно так:
    if iter_counter < 1:
        df.to_csv('example_changed.csv', index=False)
    else:
        df.to_csv('example_changed.csv', index=False, 
                  mode='a', header=False)
"""