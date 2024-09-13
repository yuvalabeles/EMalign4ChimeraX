import math
# import os
from chimerax.map_fit import fitcmd
from chimerax.map_fit.fitmap import map_overlap_and_correlation as calc_overlap_and_corr
from chimerax.map_data import arraygrid
import numpy as np
from chimerax.core.errors import UserError
from . import align_volumes_3d, reshift_vol, fastrotate3d
from .common_finufft import cryo_downsample, cryo_crop

# from . import read_write as mrc
# import mrcfile


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


def emalign(session, ref_map, query_map, downsample=64, projections=30, show_log=True, show_param=True, refine=False):
    log = session.logger

    # Calculate overlap and correlation (calculated using only data above contour level from first map):
    print_to_log(log, "Stats before alignment with EMalign:")
    overlap, corr, corr_m = calc_overlap_and_corr(query_map, ref_map, False)
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
        # Create a copy of the ref_vol to run on:
        ref_vol_copy = ref_vol.copy()

        # query_vol has the bigger pixel size ---> downsample ref_vol to N_ref_ds and then crop it to N_query:
        N_ref_ds = math.floor(N_ref * (pixel_ref / pixel_query))

        # Now we downsample ref_vol from N_ref to N_ref_ds:
        print_to_log(log, f"---> Size to downsample reference map to = {N_ref_ds}", show_log=show_log)
        ref_vol_ds = cryo_downsample(ref_vol_copy, (N_ref_ds, N_ref_ds, N_ref_ds))
        ref_vol_cropped = cryo_crop(ref_vol_ds.copy(), (N_query, N_query, N_query))
        print_to_log(log, f"---> Shape of ref_vol_cropped: {ref_vol_cropped.shape}", show_log=show_log)

        opt.align = [False]

        # At this point both volumes are the same dimension
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol_cropped, query_vol,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session)

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

    elif pixel_ref > pixel_query:  # ref_vol has the bigger pixel size and the smaller volume size
        # Create a copy of the query_vol to run align_volumes on:
        query_vol_copy = query_vol.copy()

        N_query_ds = math.floor(N_query * (pixel_query / pixel_ref))
        print_to_log(log, f"---> Size to downsample query map to = {N_query_ds}", show_log=show_log)

        # Now we downsample the copy of query_vol from N_query to N_query_ds and crop to get the same dimensions:
        query_vol_copy_ds = cryo_downsample(query_vol_copy, (N_query_ds, N_query_ds, N_query_ds))
        query_vol_copy_cropped = cryo_crop(query_vol_copy_ds.copy(), (N_ref, N_ref, N_ref))
        print_to_log(log, f"---> Shape of query_vol_cropped: {query_vol_copy_cropped.shape}", show_log=show_log)
        query_vol_copy_cropped = np.ascontiguousarray(query_vol_copy_cropped)

        opt.align = [False]

        bestR, bestdx, reflect, vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol_copy_cropped,
                                                                             opt=opt,
                                                                             show_log=show_log,
                                                                             session=session)

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

    else:
        # pixel_ref == pixel_query ---> no need to downsample anything
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session)

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
    overlap, corr, corr_m = calc_overlap_and_corr(query_map, ref_map, False)
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
