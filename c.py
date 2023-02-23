import pandas as pd
import numpy as np

base_file= 'D:/dataset/lu/data/nuclei/allf_test1006.xlsx'
df=pd.read_excel(base_file)
# df=pd.DataFrame(df)
col_name=df.columns.tolist()
print(col_name)
print(len(col_name))

# new_col_name=list(map(lambda x:'tub_'+x if x!='img_name' else x,col_name))      #新的列名
# df.columns=new_col_name                                                         #改为新的列名
# print(new_col_name)
# print(len(new_col_name))
# df.to_excel(base_file,index=None)


