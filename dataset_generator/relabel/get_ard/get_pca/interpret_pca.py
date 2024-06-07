# 3rd party modules
import xarray as xr


def pca_model_to_dataset(model, reshaped):
    pca_ds = xr.Dataset(
        data_vars = dict(
            components = (('component', 'feature'), model.components_),
            explained_variance = (('component'), model.explained_variance_),
            explained_variance_ratio = (('component'), model.explained_variance_ratio_),
            singular_values = (('component'), model.singular_values_),
        ),
        coords = reshaped.coords,
    ).unstack('feature')
    pca_ds = pca_ds.assign_coords(component = pca_ds.component)
    return pca_ds
