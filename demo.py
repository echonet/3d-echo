from extractor import ViewExtractor

view_extractor = ViewExtractor(original_weights=True)
view_extractor.extract_views('sample_3d.dcm', 'sample_output')