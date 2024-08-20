import math
# import os

from chimerax.map_fit import fitcmd
from chimerax.map_data import arraygrid
import numpy as np
from chimerax.core.errors import UserError
from . import align_volumes_3d, reshift_vol, fastrotate3d
from .common_finufft import cryo_downsample, cryo_crop


import mrcfile


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


def emalign(session, ref_map, query_map, downsample=64, projections=30, show_log=True, show_param=True, refine=True):
    log = session.logger

    # Save original parameters of ref_map: {origin, step, cell_angles, rotation, symmetries, name}
    grid_ref_map_data = ref_map.data
    ref_map_origin = grid_ref_map_data.origin
    ref_map_step = grid_ref_map_data.step  # this is the voxel size
    ref_map_cell_angles = grid_ref_map_data.cell_angles
    ref_map_rotation = grid_ref_map_data.rotation
    ref_map_symmetries = grid_ref_map_data.symmetries
    ref_map_name = grid_ref_map_data.name

    # Save original parameters of query_map: {origin, step, cell_angles, rotation, symmetries, name}
    grid_query_map_data = query_map.data
    query_map_origin = grid_query_map_data.origin
    query_map_step = grid_query_map_data.step  # this is the voxel size
    query_map_cell_angles = grid_query_map_data.cell_angles
    query_map_rotation = grid_query_map_data.rotation
    query_map_symmetries = grid_query_map_data.symmetries
    query_map_name = grid_query_map_data.name

    grid_ref_map = ref_map.full_matrix().T
    grid_query_map = query_map.full_matrix().T

    ref_vol = np.ascontiguousarray(grid_ref_map)
    query_vol = np.ascontiguousarray(grid_query_map)

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

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = projections
    opt.downsample = downsample

    # Grid size:
    N_ref = np.shape(ref_vol)[0]
    N_query = np.shape(query_vol)[0]

    # Pixel:
    pixel_ref = ref_map.data.step[0]
    pixel_query = query_map_step[0]

    # # Dimensions:
    # s1 = pixel_ref * N_ref
    # s2 = pixel_query * N_query

    if pixel_query > pixel_ref:
        # query_vol has the bigger pixel size ---> calculate N_ref_ds = N_ref*(pixel_ref/pixel_query) ---> downsample ref_vol to N_ref_ds
        N_ref_ds = math.floor(N_ref * (pixel_ref / pixel_query))
        # s1_ds = pixel_ref * N_ref_ds

        # now we downsample ref_vol from N_ref to N_ref_ds:
        log.info(f"size to downsample reference map to = {N_ref_ds}")
        ref_vol_ds = cryo_downsample(ref_vol, (N_ref_ds, N_ref_ds, N_ref_ds))
        # crop_out = abs(N_ref_ds - N_ref)  # this is an integer, we need a 3d array of crop_out
        crop_out_3d = (N_query, N_query, N_query)
        ref_vol_cropped = cryo_crop(ref_vol_ds, crop_out_3d)
        log.info(f'shape of ref_vol_cropped: {ref_vol_cropped.shape}')

        cropped_ref_map_grid_data = arraygrid.ArrayGridData(ref_vol_cropped.T, origin=ref_map_origin,
                                                            step=query_map_step,
                                                            cell_angles=ref_map_cell_angles,
                                                            rotation=ref_map_rotation,
                                                            symmetries=ref_map_symmetries, name=ref_map_name)

        # Replace the data in the original query_map:
        ref_map.replace_data(cropped_ref_map_grid_data)

        grid_ref_map = ref_map.full_matrix().T
        ref_vol = np.ascontiguousarray(grid_ref_map)

        # at this point both volumes are the same dimension
        # opt.p = pixel_query / pixel_ref
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   show_param=show_param,
                                                                                   session=session)

        # ref_map_step = query_map_step
    elif pixel_ref > pixel_query:
        # ref_vol has the bigger pixel size ---> calculate N_query_ds = N_query*(pixel_query/pixel_ref) ---> downsample query_vol to N_query_ds
        N_query_ds = math.floor(N_query * (pixel_query / pixel_ref))
        # s2_ds = pixel_query * N_query_ds

        # cropout = N_query_ds - N_ref  # this is how much we'll crop out after the downsampling
        # N_query_ds = N_query - cropout
        # log.info(f"size to crop first the query map to = {N_query_ds}")
        # crop_out_3d = (N_query_ds, N_query_ds, N_query_ds)
        # query_vol_cropped = cryo_crop(query_vol, crop_out_3d)
        # log.info(f'shape of query_vol_cropped **: {query_vol_cropped.shape}')
        # query_vol_cropped = cryo_downsample(query_vol_cropped, (N_ref, N_ref, N_ref))

        ################################################################################################################
        # now we downsample query_vol from N_query to N_query_ds and crop to get the same dimensions:
        log.info(f"size to downsample query map to = {N_query_ds}")
        query_vol_ds = cryo_downsample(query_vol.copy(), (N_query_ds, N_query_ds, N_query_ds))
        crop_out_3d = (N_ref, N_ref, N_ref)
        query_vol_cropped = cryo_crop(query_vol_ds.copy(), crop_out_3d)
        log.info(f'shape of query_vol_cropped **: {query_vol_cropped.shape}')
        # Save the new cropped, downsampled map:
        # name_to_save = "C:\\users\\yuval\\crp_ds_" + query_map_name
        # query_map_step = ref_map_step
        # mrcfile.write(name_to_save, query_vol_cropped.T, overwrite=True, voxel_size=query_map_step)
        # reopened_query_map = mrcfile.open(name_to_save).data
        # query_vol_cropped = np.ascontiguousarray(reopened_query_map)
        # log.info(f'shape of REOPENED query_vol_cropped **: {query_vol_cropped.shape}')

        ################################################################################################################

        # query_map.show(show=False)
        # query_map_step = ref_map_step
        # cropped_query_map_grid_data = arraygrid.ArrayGridData(query_vol_cropped.T, origin=query_map_origin,
        #                                                       step=query_map_step,
        #                                                       cell_angles=query_map_cell_angles,
        #                                                       rotation=query_map_rotation,
        #                                                       symmetries=query_map_symmetries, name=query_map_name)

        # So far we have a cropped, downsampled map that matches the other volume's dimensions,
        # now we need to create an additional copy of the ORIGINAL map and replace its data with the cropped data.

        # Create new map copy:
        # query_map_copy = query_map.copy()
        # Replace the data in the copied query_map:
        # query_map.replace_data(cropped_query_map_grid_data)
        # Reopen the map (the cropped anddownsampled one):
        # grid_query_map = query_map.full_matrix().T
        # query_vol = np.ascontiguousarray(grid_query_map)

        # log.info(f'shape of query_vol_cropped AFTER: {query_vol.shape}')
        # log.info(f"type of input aligned volume: {query_vol.dtype}")

        # at this point both volumes are the same dimension
        # opt.p = pixel_query / pixel_ref
        ################################################################################################################
        opt.align = [False, query_vol, N_query_ds, crop_out_3d, N_ref]

        bestR, bestdx, reflect, vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol_cropped,
                                                                             opt=opt,
                                                                             show_log=show_log,
                                                                             show_param=show_param,
                                                                             session=session)
        # log.info(f"type of output aligned volume: {query_vol_aligned.dtype}")

        # Get query_vol to be the original uncropped map:
        # grid_query_map = query_map.full_matrix().T
        # query_vol = np.ascontiguousarray(grid_query_map)

        if show_log:
            log.info('---> Translating original volumes (in emalign_cmd.py)')
        query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)

        bestdx = (pixel_ref / pixel_query) * bestdx
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
        else:
            query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)

        if show_log:
            log.info('---> Computing correlations of original volumes')
        # bestcorr = np.mean(np.corrcoef(ref_vol.ravel(), query_vol_aligned.ravel(), rowvar=False)[0, 1:])

        if show_param:
            log.info('Estimated rotation:')
            log.info(f'[[{bestR[0, 0]:.3f} {bestR[0, 1]:.3f} {bestR[0, 2]:.3f}],')
            log.info(f'[{bestR[1, 0]:.3f} {bestR[1, 1]:.3f} {bestR[1, 2]:.3f}]')
            log.info(f'[{bestR[2, 0]:.3f} {bestR[2, 1]:.3f} {bestR[2, 2]:.3f}]]')
            log.info(f'Estimated translations: [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]')
            # log.info(f'Correlation between original aligned volumes is {bestcorr:.4f}')

        query_vol_aligned = query_vol_aligned.astype(np.float32)
        ################################################################################################################
        # query_vol_aligned = query_vol

    # if pixel_ref != pixel_query:
    #     opt.align = False
    #
    #     if pixel_ref > pixel_query:
    #         pixel_factor = pixel_query / pixel_ref
    #
    #         # ref_vol has the bigger pixel size ---> downsample query_vol to N_query_ds
    #         N_query_ds = math.floor(N_query * pixel_factor)
    #         log.info(
    #             f"Volumes have different dimensions, so we downsample the query map from {N_query} to {N_query_ds}")
    #
    #         # Now we downsample query_vol from N_query to N_query_ds:
    #         query_vol_ds = cryo_downsample(query_vol, (N_query_ds, N_query_ds, N_query_ds))
    #         query_vol_cropped = cryo_crop(query_vol_ds, (N_ref, N_ref, N_ref))
    #         log.info(f"Then we crop the query map from {N_query_ds} to {N_ref}")
    #
    #         query_map_step = ref_map_step
    #         cropped_query_map_grid_data = arraygrid.ArrayGridData(query_vol_cropped.T, origin=query_map_origin,
    #                                                               step=query_map_step,
    #                                                               cell_angles=query_map_cell_angles,
    #                                                               rotation=query_map_rotation,
    #                                                               symmetries=query_map_symmetries, name=query_map_name)
    #         # Replace the data in the original query_map:
    #         query_map.replace_data(cropped_query_map_grid_data)
    #         grid_query_map = query_map.full_matrix().T
    #         query_vol = np.ascontiguousarray(grid_query_map)
    #
    #         # At this point both volumes are the same dimension
    #         bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
    #                                                                                    opt=opt,
    #                                                                                    show_log=show_log,
    #                                                                                    show_param=show_param,
    #                                                                                    session=session)
    #     else:
    #         pixel_factor = pixel_ref / pixel_query
    #
    #         # query_vol has the bigger pixel size ---> downsample ref_vol to N_ref_ds
    #         N_ref_ds = math.floor(N_ref * pixel_factor)
    #         log.info(
    #             f"Volumes have different dimensions, so we downsample the reference map from {N_ref} to {N_ref_ds}")
    #
    #         # Now we downsample ref_vol from N_ref to N_ref_ds:
    #         ref_vol_ds = cryo_downsample(ref_vol, (N_ref_ds, N_ref_ds, N_ref_ds))
    #         ref_vol_cropped = cryo_crop(ref_vol_ds, (N_query, N_query, N_query))
    #         log.info(f"Then we crop the reference map from {N_ref_ds} to {N_query}")
    #
    #         cropped_ref_map_grid_data = arraygrid.ArrayGridData(ref_vol_cropped.T, origin=ref_map_origin,
    #                                                             step=query_map_step,
    #                                                             cell_angles=ref_map_cell_angles,
    #                                                             rotation=ref_map_rotation,
    #                                                             symmetries=ref_map_symmetries, name=ref_map_name)
    #         # Replace the data in the original ref_map:
    #         ref_map.replace_data(cropped_ref_map_grid_data)
    #         grid_ref_map = ref_map.full_matrix().T
    #         ref_vol = np.ascontiguousarray(grid_ref_map)
    #
    #         # At this point both volumes are the same dimension
    #         bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
    #                                                                                    opt=opt,
    #                                                                                    show_log=show_log,
    #                                                                                    show_param=show_param,
    #                                                                                    session=session)
    #         ref_map.replace_data(grid_ref_map_data)
    #
    #     pixel_factor = 1 / pixel_factor
    #     log.info("Aligning volumes AFTER:")
    #     if show_log:
    #         log.info('---> Translating original volumes')
    #     query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)
    #     bestdx = pixel_factor * bestdx
    #     if (np.round(bestdx) == bestdx).all():
    #         # Use fast method:
    #         query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
    #     else:
    #         query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)
    #
    #     if show_log:
    #         log.info('---> Computing correlations of original volumes')
    #     bestcorr = np.mean(np.corrcoef(ref_vol.ravel(), query_vol_aligned.ravel(), rowvar=False)[0, 1:])
    #
    #     if show_param:
    #         log.info('Estimated rotation:')
    #         log.info(f'[[{bestR[0, 0]:.3f} {bestR[0, 1]:.3f} {bestR[0, 2]:.3f}],')
    #         log.info(f'[{bestR[1, 0]:.3f} {bestR[1, 1]:.3f} {bestR[1, 2]:.3f}]')
    #         log.info(f'[{bestR[2, 0]:.3f} {bestR[2, 1]:.3f} {bestR[2, 2]:.3f}]]')
    #         log.info(f'Estimated translations: [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]')
    #         log.info(f'Correlation between original aligned volumes is {bestcorr:.4f}')
    #
    #     query_vol_aligned = query_vol_aligned.astype(np.float32)
    else:
        # pixel_ref == pixel_query ---> no need to downsample anything
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol, query_vol,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   show_param=show_param,
                                                                                   session=session)

    # if ref_vol.shape[0] != query_vol.shape[0]:
    #     raise UserError("Input volumes must be of same dimensions")
    #
    # # Run:
    # bestR, bestdx, reflect, query_vol_aligned, bestcorr = align_volumes_3d.align_volumes(ref_vol, query_vol, opt=opt,
    #                                                                                      show_log=show_log,
    #                                                                                      show_param=show_param,
    #                                                                                      session=session)

    # Hide the display of query_map (query_vol) prior to the alignment:
    query_map.show(show=False)

    # Create GridData object with aligned query_vol but with the original query_map parameters:
    aligned_map_grid_data = arraygrid.ArrayGridData(query_vol_aligned.T, origin=query_map_origin,
                                                    step=query_map_step,
                                                    cell_angles=query_map_cell_angles, rotation=query_map_rotation,
                                                    symmetries=query_map_symmetries, name=query_map_name)

    # aligned_map_grid_data = arraygrid.ArrayGridData(query_vol_aligned, origin=query_map_origin, step=query_map_step)

    # Replace the data in the original query_map:
    query_map.replace_data(aligned_map_grid_data)

    # fitmap query_map inMap ref_map:
    # if show_log:
    #     log.info("---> Using fitmap to perform final refinement")
    if refine:
        fitcmd.fit_map_in_map(query_map, ref_map, metric='correlation', envelope=True, zeros=False, shift=True,
                              rotate=True,
                              move_whole_molecules=True, map_atoms=None, max_steps=2000, grid_step_min=0.01,
                              grid_step_max=0.5)

    # Show the query_map (aligned):
    query_map.show(show=True)

    # curr_dir = os.getcwd()
    # log.info(f"Current working directory: {curr_dir}")

    # # log.info(f'shape of original query_vol: {query_vol.shape}')
    # mrcfile.write("C:\\users\\yuval\\original_" + query_map_name, query_vol.T, overwrite=True,
    #               voxel_size=query_map_step)
    # # log.info(f'shape of COPY query_vol: {query_vol_aligned.shape}')
    # mrcfile.write("C:\\users\\yuval\\cropped_" + query_map_name, query_vol_aligned.T, overwrite=True,
    #               voxel_size=query_map_step)

    # ref_map.set_step(query_map_step)
    # query_map.set_step(ref_map_step)

    if show_log:
        log.info("* Alignment completed *")
