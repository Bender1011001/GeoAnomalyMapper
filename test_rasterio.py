try:
    import rasterio
    print('Rasterio imported successfully:', rasterio.__version__)
except Exception as e:
    print('Import failed:', str(e))
    import traceback
    traceback.print_exc()