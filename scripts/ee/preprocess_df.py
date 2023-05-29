import pandas as pd


def get_dates(dataframe: pd.DataFrame, tile: str, dates: dict) -> list:
    """Get a list of dates from a dataframe"""
    for column in dataframe.columns.tolist():
        if column.startswith('20'):
            if column[:8] not in dates[tile]:
                dates[tile].append(column[:8])
    return dates[tile]


def rename_bands(dataframe: pd.DataFrame, tile: str, common_dates: set) -> pd.DataFrame:
    """
    Keep only bands that are in common_dates and rename bands
    """
    band_names = {tile: list(),
                  f'{tile}_drop': list()}
    for column in dataframe.columns:
        if column[:8] in common_dates:
            band_names.get(tile).append(column)
        elif column.startswith('20'):
            band_names.get(f'{tile}_drop').append(column)
    dataframe = dataframe.drop(columns=band_names[f'{tile}_drop'])
    rename_df = dict()
    for band in band_names.get(tile):
        rename_df[band] = band[:8] + band[-4:]
    return dataframe.rename(columns=rename_df)


def main():
    df_30stc = pd.read_csv('../../data/sampled30STC.csv', dtype='unicode') \
        .drop(columns=['system:index', '.geo'])
    df_29squ = pd.read_csv('../../data/sampled29SQU.csv', dtype='unicode') \
        .drop(columns=['system:index', '.geo'])

    dates = {'29SQU': [], '30STC': []}

    dates['30STC'] = get_dates(df_30stc, '30STC', dates)
    dates['29SQU'] = get_dates(df_29squ, '29SQU', dates)

    common_dates = set(dates['29SQU']).intersection(dates['30STC'])

    df_30stc = rename_bands(df_30stc, '30STC', common_dates)
    df_29squ = rename_bands(df_29squ, '29SQU', common_dates)

    df = pd.concat([df_30stc, df_29squ])
    df = df.reset_index().drop('index', axis=1)

    # cultures = ['ble tendre', 'orge', 'ble dur', 'avoine',
    #             'colza', 'grenadier', 'oranger',
    #             'pois chiche', 'feverole',
    #             'melon', 'oignon']

    # df = df.query('culture.isin(@cultures)')

    band_cols = df.drop(columns=['culture', 'filiere', 'TRAIN']).columns

    for band_col in band_cols:
        df[band_col] = df[band_col].astype('int16')

    df['TRAIN'] = df['TRAIN'].astype('int8')

    df.to_parquet('../../data/ee_sampled_pts_df_2021.parquet')


if __name__ == '__main__':
    main()
