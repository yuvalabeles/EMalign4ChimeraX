import math
import time
import numpy as np
from chimerax.map_fit import fitcmd
from chimerax.map_fit.fitmap import map_overlap_and_correlation as calculate_stats
from chimerax.map_data import arraygrid
from chimerax.core.errors import UserError
from . import align_volumes_3d, reshift_vol, fastrotate3d
from .common_finufft import cryo_downsample, cryo_crop
from .utils import fuzzy_mask


def register_emalign_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import IntArg
    from chimerax.map.mapargs import MapArg

    emalign_desc = CmdDesc(
        required=[
            ('ref_map', MapArg),
        ],
        keyword=[
            ('query_map', MapArg)

        ],
        optional=[
            ('downsample', IntArg),
            ('projections', IntArg)
        ],
        required_arguments=['ref_map', 'query_map'],
        synopsis='Perform EM-alignment of two density maps'
    )
    register('volume emalign', emalign_desc, emalign, logger=logger)


def emalign(session, ref_map, query_map, downsample=64, projections=50, mask=False, show_log=True, show_param=True, refine=False):
    log = session.logger

    # TODO add the option to choose whether the correaltion is computed above threshold (like in Fit in Map)
    # Calculate overlap and correlation (calculated using only data above contour level from first map):
    print_to_log(log, "\nStats before alignment with EMalign:")
    overlap, corr, corr_m = calculate_stats(query_map, ref_map, True)
    print_to_log(log, f"correlation = {corr:.4f}, correlation about mean = {corr_m:.4f}, overlap = {overlap:.3f}\n")

    ref = ref_map.data
    query = query_map.data

    # Save original parameters of ref_map and query_map: {origin, step, cell_angles, rotation, symmetries, name}
    ref_dict = {}
    query_dict = {}
    keys = ["origin", "step", "cell_angles", "rotation", "symmetries", "name"]
    ref_map_values = [ref.origin, ref.step, ref.cell_angles, ref.rotation, ref.symmetries, ref.name]
    query_map_values = [query.origin, query.step, query.cell_angles, query.rotation, query.symmetries, query.name]
    for i in range(len(keys)):
        ref_dict[keys[i]] = ref_map_values[i]
        query_dict[keys[i]] = query_map_values[i]

    grid_ref_map = ref_map.full_matrix()
    grid_query_map = query_map.full_matrix()

    ref_vol = np.ascontiguousarray(grid_ref_map)
    query_vol = np.ascontiguousarray(grid_query_map)

    ref_vol, query_vol = validate_input(ref_vol, query_vol)

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = projections
    opt.downsample = downsample

    # Grid size:
    N_ref = np.shape(ref_vol)[0]
    N_query = np.shape(query_vol)[0]

    # Pixel:
    pixel_ref = ref_dict.get("step")[0]
    pixel_query = query_dict.get("step")[0]

    if pixel_query > pixel_ref:
        t1 = time.perf_counter()

        # Create a copy of the ref_vol to run on:
        ref_vol_copy = ref_vol.copy()
        query_vol_copy = query_vol.copy()

        if mask:
            optimal_radius = find_optimal_radius(ref_vol)
            r0_factor = optimal_radius / N_ref

            m1 = fuzzy_mask([N_ref, N_ref, N_ref], dtype=np.float32, r0=r0_factor * N_ref)
            m2 = fuzzy_mask([N_query, N_query, N_query], dtype=np.float32, r0=r0_factor * N_query)

            ref_vol_copy = ref_vol_copy * m1
            query_vol_copy = query_vol_copy * m2

        # query_vol has the bigger pixel size ---> downsample ref_vol to N_ref_ds and then crop it to N_query:
        N_ref_ds = math.floor(N_ref * (pixel_ref / pixel_query))

        # Now we downsample ref_vol from N_ref to N_ref_ds:
        print_to_log(log, f"---> Size to downsample reference map to = {N_ref_ds}", show_log=show_log)
        ref_vol_ds = cryo_downsample(ref_vol_copy, (N_ref_ds, N_ref_ds, N_ref_ds))
        ref_vol_cropped = cryo_crop(ref_vol_ds.copy(), (N_query, N_query, N_query))
        print_to_log(log, f"---> Shape of ref_vol_cropped: {ref_vol_cropped.shape}", show_log=show_log)

        opt.align = [False]

        # At this point both volumes are the same dimension
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol_cropped,
                                                                                   query_vol_copy,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session)
        if reflect:
            print_to_log(log, "---> Flipping query volume before aligning original volumes", show_log=show_log)
            query_vol = np.flip(query_vol, axis=2)

        print_to_log(log, "---> Aligning original different sized volumes\n", show_log=show_log)

        # Rotate the volumes:
        query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)

        # Translate the volumes:
        bestdx = (pixel_query / pixel_ref) * bestdx
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
        else:
            query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)

        query_vol_aligned = query_vol_aligned.astype(np.float32)

        t2 = time.perf_counter()
        print_to_log(log, f"Aligning the volumes using EMalign took {t2 - t1:.2f} seconds\n")

    elif pixel_ref > pixel_query:  # ref_vol has the bigger pixel size and the smaller volume size
        t1 = time.perf_counter()

        # Create a copy of the query_vol to run align_volumes on:
        ref_vol_copy = ref_vol.copy()
        query_vol_copy = query_vol.copy()

        if mask:
            print_to_log(log, f"---> Using masking to clean volumes for more accurate alignment (90% of volumes)", show_log=show_log)
            optimal_radius = calc_3D_radius(query_vol)
            r0_factor = optimal_radius / N_query

            m1 = fuzzy_mask([N_ref, N_ref, N_ref], dtype=np.float32, r0=r0_factor * N_ref)
            m2 = fuzzy_mask([N_query, N_query, N_query], dtype=np.float32, r0=r0_factor * N_query)

            ref_vol_copy = ref_vol_copy * m1
            query_vol_copy = query_vol_copy * m2

        N_query_ds = math.floor(N_query * (pixel_query / pixel_ref))
        print_to_log(log, f"---> Size to downsample query volume to = {N_query_ds}", show_log=show_log)

        # Downsample the copy of query_vol from N_query to N_query_ds:
        query_vol_copy_ds = cryo_downsample(query_vol_copy, (N_query_ds, N_query_ds, N_query_ds))
        query_vol_copy_cropped = cryo_crop(query_vol_copy_ds.copy(), (N_ref, N_ref, N_ref))

        # Crop to get the same dimensions as ref_vol:
        print_to_log(log, f"---> Shape of cropped query volume: {query_vol_copy_cropped.shape}", show_log=show_log)
        query_vol_copy_cropped = np.ascontiguousarray(query_vol_copy_cropped)

        opt.options = [False]

        bestR, bestdx, reflect, vol_aligned = align_volumes_3d.align_volumes(ref_vol_copy,
                                                                             query_vol_copy_cropped,
                                                                             opt=opt,
                                                                             show_log=show_log,
                                                                             session=session)

        if reflect:
            print_to_log(log, "---> Flipping query volume before aligning original volumes", show_log=show_log)
            query_vol = np.flip(query_vol, axis=2)

        print_to_log(log, "---> Aligning original different sized volumes\n", show_log=show_log)

        # Rotate the volumes:
        query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)

        # Translate the volumes:
        bestdx = (pixel_ref / pixel_query) * bestdx
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
        else:
            query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)

        query_vol_aligned = query_vol_aligned.astype(np.float32)

        t2 = time.perf_counter()
        print_to_log(log, f"Aligning the volumes using EMalign took {t2 - t1:.2f} seconds\n")

    else:
        # pixel_ref == pixel_query ---> no need to downsample anything
        t1 = time.perf_counter()
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session)
        t2 = time.perf_counter()
        print_to_log(log, f"Aligning the volumes using EMalign took {t2 - t1:.2f} seconds\n")

    print_param(log, bestR, bestdx, show_param)

    # Create GridData object with aligned query_vol but with the original query_map parameters:
    aligned_map_grid_data = arraygrid.ArrayGridData(query_vol_aligned, origin=query_dict.get("origin"),
                                                    step=query_dict.get("step"),
                                                    cell_angles=query_dict.get("cell_angles"),
                                                    rotation=query_dict.get("rotation"),
                                                    symmetries=query_dict.get("symmetries"),
                                                    name=query_dict.get("name"))

    # Replace the data in the original query_map:
    query_map.replace_data(aligned_map_grid_data)

    # Calculate overlap and correlation (calculated using only data above contour level from first map):
    print_to_log(log, "Stats after applying the transformations:")
    overlap, corr, corr_m = calculate_stats(query_map, ref_map, True)
    print_to_log(log, f"correlation = {corr:.4f}, correlation about mean = {corr_m:.4f}, overlap = {overlap:.3f}\n")

    # Perform additional refinement with Fit in Map:
    if refine:
        print_to_log(log, "Used Fit in Map to perform additional refinement:")

        # Run fitmap to fit query_map in ref_map:
        fitcmd.fit_map_in_map(query_map, ref_map, metric="correlation", envelope=True, zeros=False, shift=True,
                              rotate=True, move_whole_molecules=True, map_atoms=None,
                              max_steps=2000, grid_step_min=0.01, grid_step_max=0.5, log=log)


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Helper Functions: ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def generate_points_and_distances(shape):
    x, y, z = np.indices(shape)
    center = np.array(shape) // 2
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    distances = distances.flatten()
    return points, distances


def calc_3D_radius(density_map, energy_fraction=0.90):
    shape = density_map.shape
    # print("Create a list of all points and their distances from the center")
    points, distances = generate_points_and_distances(shape)

    # Compute the energy at each point
    # print("Compute the total energy")
    energy_values = density_map.flatten() ** 2

    # Sort distances and corresponding energy values
    # print("Sort distances and corresponding energy values")
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_energy_values = energy_values[sorted_indices]

    # Compute the cumulative energy
    # print("Compute cumulative energy")
    cumulative_energy = np.cumsum(sorted_energy_values)

    # Total energy
    total_energy = cumulative_energy[-1]

    # Find the radius where cumulative energy reaches the specified fraction
    # print("Find the radius where cumulative energy reaches the specified fraction")
    target_energy = total_energy * energy_fraction
    radius_index = np.searchsorted(cumulative_energy, target_energy)

    return sorted_distances[radius_index]


def find_knee(cumsum_signal):
    """
    Find the knee point in the cumulative sum signal by calculating the maximum distance
    from the line connecting the first and last points of the signal.

    Parameters:
    - cumsum_signal (numpy.ndarray): 1D array containing the cumulative sum of the signal.

    Returns:
    - knee_index (int): Index of the knee point in the cumsum signal.
    """
    # Normalize the cumsum signal to be between 0 and 1
    norm_signal = (cumsum_signal - cumsum_signal[0]) / (cumsum_signal[-1] - cumsum_signal[0])

    # Create a straight line from the first to the last point
    line = np.linspace(0, 1, len(cumsum_signal))

    # Calculate the distance from each point in the signal to the line
    distances = np.abs(norm_signal - line)

    # Find the index of the maximum distance (the knee point)
    knee_index = np.argmax(distances)

    return knee_index


def find_centered_circle(arr, energy_fraction=0.9):
    """
    Find the radius of a centered circle that contains the specified fraction of energy of the 2D array.

    Parameters:
    - arr (numpy.ndarray): 2D array representing the data.
    - energy_fraction (float): Fraction of the total energy to be contained in the circle (default 90%).

    Returns:
    - radius (float): The radius of the circle that contains the specified fraction of energy.
    """
    # Compute total energy (sum of squares of all elements)
    total_energy = np.sum(arr ** 2)

    # Center coordinates
    center_x, center_y = np.array(arr.shape) // 2

    # Create a distance grid relative to the center
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Flatten arrays for easier sorting
    distances_flat = distances.flatten()
    energy_flat = (arr ** 2).flatten()

    # Sort by increasing distance from the center
    sorted_indices = np.argsort(distances_flat)
    sorted_energy = energy_flat[sorted_indices]
    sorted_distances = distances_flat[sorted_indices]

    # Cumulative energy sum
    cumulative_energy = np.cumsum(sorted_energy)

    # Find the distance at which cumulative energy reaches the desired fraction
    target_energy = energy_fraction * total_energy
    radius_idx = np.searchsorted(cumulative_energy, target_energy)

    # Get the corresponding radius
    radius = sorted_distances[radius_idx]

    return radius


def find_optimal_radius(vol):
    # Iterate over slices in the 3d array and find radius for each axis:
    n = vol.shape[0]
    r_x_list = []
    r_y_list = []
    r_z_list = []
    for i in range(n):
        vol_2d_yz = vol[i]
        r_x_i = find_centered_circle(vol_2d_yz)
        r_x_list.append(r_x_i)

        vol_2d_xz = vol[:, i, :]
        r_y_i = find_centered_circle(vol_2d_xz)
        r_y_list.append(r_y_i)

        vol_2d_xy = vol[:, :, i]
        r_z_i = find_centered_circle(vol_2d_xy)
        r_z_list.append(r_z_i)

    r_x_mean = np.mean(r_x_list)
    r_y_mean = np.mean(r_y_list)
    r_z_mean = np.mean(r_z_list)

    optimal_radius = max(r_x_mean, r_y_mean, r_z_mean)

    return optimal_radius


def validate_input(ref_vol, query_vol):
    # Handle the case where ref_vol is 4D:
    if (ref_vol.ndim == 4) and (ref_vol.shape[-1] == 1):
        ref_vol = np.squeeze(ref_vol)
    elif ref_vol.ndim != 3:
        raise UserError("Volumes must be three-dimensional or four-dimensional with singleton first dimension ")

    # Handle the case where query_vol is 4D:
    if (query_vol.ndim == 4) and (query_vol.shape[-1] == 1):
        query_vol = np.squeeze(query_vol)
    elif query_vol.ndim != 3:
        raise UserError("Volumes must be three-dimensional or four-dimensional with singleton first dimension ")

    if not ((ref_vol.shape[1] == ref_vol.shape[0]) and (ref_vol.shape[2] == ref_vol.shape[0])
            and (ref_vol.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    if not ((query_vol.shape[1] == query_vol.shape[0]) and (query_vol.shape[2] == query_vol.shape[0])
            and (query_vol.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    return ref_vol, query_vol


def print_param(log, bestR, bestdx, show_param):
    if show_param:
        log.info('Estimated Rotation:')
        log.info(f'[[{bestR[0, 0]:.3f} {bestR[0, 1]:.3f} {bestR[0, 2]:.3f}],')
        log.info(f'[{bestR[1, 0]:.3f} {bestR[1, 1]:.3f} {bestR[1, 2]:.3f}]')
        log.info(f'[{bestR[2, 0]:.3f} {bestR[2, 1]:.3f} {bestR[2, 2]:.3f}]]')
        log.info(f'Estimated Translations:\n [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]\n')


def print_to_log(log, msg, show_log=True):
    if show_log:
        log.info(msg)
