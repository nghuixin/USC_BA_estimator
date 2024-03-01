import numpy as np
import os

def processmgz(brains, voxel_size=(128, 128, 128)):
    """
    Process brain images to extract significant subvolumes with specified voxel sizes.

    Parameters:
    - brains: A list of brain images.
    - voxel_size: A tuple indicating the desired dimensions (width, height, depth) of the voxel grid.

    Returns:
    - data_new: A list of processed brain images with the specified voxel dimensions.
    - coord: A list of coordinates representing the center of significant subvolumes in each brain image.
    """
    
    X = np.asarray(brains)
    coord = []
    voxel_width, voxel_height, voxel_depth = voxel_size
    for i in range(X.shape[0]):
        buf = X[i, :, :, :]
        buf = buf.reshape(voxel_size)
        xmin = xmax = ymin = ymax = zmin = zmax = 0
        for xm in range(voxel_width):
            if np.sum(buf[xm, :, :]) > 50:
                xmin = xm
                break
        for xm in range(voxel_width):
            if np.sum(buf[voxel_width - 1 - xm, :, :]) > 50:
                xmax = voxel_width - 1 - xm
                break
        for ym in range(voxel_height):
            if np.sum(buf[:, ym, :]) > 50:
                ymin = ym
                break
        for ym in range(voxel_height):
            if np.sum(buf[:, voxel_height - 1 - ym, :]) > 50:
                ymax = voxel_height - 1 - ym
                break
        for zm in range(voxel_depth):
            if np.sum(buf[:, :, zm]) > 50:
                zmin = zm
                break
        for zm in range(voxel_depth):
            if np.sum(buf[:, :, voxel_depth - 1 - zm]) > 50:
                zmax = voxel_depth - 1 - zm
                break
        td = [abs(xmax + xmin) / 2, abs(ymax + ymin) / 2, abs(zmax + zmin) / 2]
        coord.append(td)

    x_range, y_range, z_range = voxel_size
    data_new = []
    for i in range(X.shape[0]):
        buf = X[i, :, :, :]
        co = coord[i]
        buf = buf.reshape(voxel_size)
        if (co[0] > (x_range / 2) and co[0] < (voxel_width - (x_range / 2)) and
            co[1] > (y_range / 2) and co[1] < (voxel_height - (y_range / 2)) and
            co[2] > (z_range / 2) and co[2] < (voxel_depth - (z_range / 2))):
            data_new.append(buf[
                int(co[0] - (x_range / 2)):int(co[0] + (x_range / 2)),
                int(co[1] - (y_range / 2)):int(co[1] + (y_range / 2)),
                int(co[2] - (z_range / 2)):int(co[2] + (z_range / 2)),
            ])
    data_new = np.expand_dims(data_new, axis=4)  # Reshape the brain MRI from 3D to 4D for consistency with expected input formats
    return data_new, coord
