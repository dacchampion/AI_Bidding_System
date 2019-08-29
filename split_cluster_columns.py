import pandas as pd

df = pd.read_csv("~/Downloads/broadterms_cpcOnly_120_desktop_clustered (1).csv")

clustered_kws = df['Clustered Words'].apply(lambda x: x.split(','))

clusters_df = pd.DataFrame(columns= ['kwd', 'clusterd_kws'])

i, j = 0, 0
for kwd in df['Unnamed: 0'].values:
    for cw in clustered_kws[j]:
        clusters_df.set_value(i, 'kwd', kwd)
        clusters_df.set_value(i, 'clusterd_kws', cw)
        i += 1
    j += 1

clusters_df.to_csv("~/Downloads/broadterms_cpcOnly_120_desktop_clustered.csv", index=False)