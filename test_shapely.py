try:
    from shapely.geometry import shape, box
    print('Shapely imported successfully:', shapely.__version__)
except Exception as e:
    print('Shapely import failed:', str(e))
    import traceback
    traceback.print_exc()