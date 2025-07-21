import tabula
dfs = tabula.read_pdf("utils/qt97mpntable-1-2.pdf", pages='all', multiple_tables=True)

mpn_dict = {}
df1 = dfs[0]
df2 = dfs[1]

for df in [df1, df2]:
    df_mpn = df.drop(columns=['Positive'])
    
    df_mpn.index = df_mpn.index.astype(int)
    df_mpn.columns = df_mpn.columns.astype(int)

    for large in df_mpn.index:
        for small in df_mpn.columns:
            mpn_dict[(large, small)] = df_mpn.at[large, small]


with open("utils/mpn_table.py", "w") as f:
    f.write("mpn_dict = {\n")
    for key, value in mpn_dict.items():
        f.write(f"    {key}: {repr(value)},\n")
    f.write("}\n")