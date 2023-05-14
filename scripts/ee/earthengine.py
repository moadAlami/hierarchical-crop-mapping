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
                       properties=['culture', 'filiere', 'TRAIN'],
                       scale=10,
                       geometries=True)


def main():
    # ee.Authenticate()
    ee.Initialize()

    poly = ee.FeatureCollection('users/mouad_alami/PhD/gharb_2021_plots_wgs')

    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(poly) \
        .filterDate('2020-12-20', '2021-05-20') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))

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
