import ee


def sample_regions(col: ee.ImageCollection, poly: ee.FeatureCollection) -> ee.FeatureCollection:
    """
    Create a new feature collection with sampled pixel values using a polygon layer
    """
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    new_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10']

    return col.select(bands).map(lambda image: image.rename(new_bands)) \
        .toBands() \
        .sampleRegions(collection=poly,
                       properties=['culture', 'filiere', 'TRAIN', 'G_TRAIN'],
                       scale=10,
                       geometries=True)


def main():
    # ee.Authenticate()
    ee.Initialize()

    poly = ee.FeatureCollection('users/mouad_alami/PhD/gharb_2021_plots_wgs')

    filterDates = ee.Filter.Or(
        ee.Filter.date('2020-12-22', '2020-12-23'),
        ee.Filter.date('2020-12-27', '2020-12-28'),
        ee.Filter.date('2021-01-03', '2021-01-04'),
        ee.Filter.date('2021-01-13', '2021-01-14'),
        ee.Filter.date('2021-01-16', '2021-01-17'),
        ee.Filter.date('2021-01-18', '2021-01-19'),
        ee.Filter.date('2021-01-26', '2021-01-27'),
        ee.Filter.date('2021-02-15', '2021-02-16'),
        ee.Filter.date('2021-03-14', '2021-03-15'),
        ee.Filter.date('2021-03-22', '2021-03-23'),
        ee.Filter.date('2021-03-24', '2021-03-25'),
        ee.Filter.date('2021-04-13', '2021-04-14'),
        ee.Filter.date('2021-04-18', '2021-04-19'),
        ee.Filter.date('2021-04-23', '2021-04-24'),
        ee.Filter.date('2021-05-06', '2021-05-07'),
        ee.Filter.date('2021-05-13', '2021-05-14'),
        ee.Filter.date('2021-05-16', '2021-05-17'),
        ee.Filter.date('2021-05-18', '2021-05-19'),
    )

    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(poly) \
        .filter(filterDates)

    s2_29_squ = s2.filter(ee.Filter.eq('MGRS_TILE', '29SQU'))
    s2_30_stc = s2.filter(ee.Filter.eq('MGRS_TILE', '30STC'))

    img_cols = [s2_29_squ, s2_30_stc]
    for img_col in img_cols:
        sampled = sample_regions(img_col, poly)

        filename = f'sampled{img_col.first().get("MGRS_TILE").getInfo()}'

        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=filename,
            fileNamePrefix=filename,
            fileFormat='CSV'
        )
        task.start()


if __name__ == '__main__':
    main()
