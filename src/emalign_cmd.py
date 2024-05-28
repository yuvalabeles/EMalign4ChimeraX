# import chimerax.core
from chimerax.map_fit import fitcmd
from chimerax.map_data import arraygrid
# from chimerax.map_data import griddata
import numpy as np
from chimerax.core.errors import UserError
# from mrcfile import mrcfile

# import chimerax.log
from . import align_volumes_3d


# import mrcfile
# from chimerax.map_data import mrc


#########
# Steps:
#########
#   - get two maps (volume data) from user (through ChimeraX), and additional args (optional)
#   - put the maps into variables using ChimeraX functions (so the maps are objects of Volume, Model, etc.)
#   - convert the maps to the format used in the emalign existing code
#   - align the volumes
#   - display the aligned map in the session
#   - hide the original (not-aligned) query map if hideMap == True


def register_emalign_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import BoolArg, IntArg
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
            ('projections', IntArg),
            ('hide_map', BoolArg)
        ],
        required_arguments=['ref_map', 'query_map'],
        synopsis='Perform EM-alignment of two density maps'
    )
    register('volume emalign', emalign_desc, emalign, logger=logger)


def emalign(session, ref_map, query_map, downsample=64, projections=30):
    log = session.logger

    # chimerax.map.volume.save_map(session, "temp_vol_1", "mrc", models=ref_map)
    # chimerax.map.volume.save_map(session, "temp_vol_2", "mrc", models=query_map)

    # grid_ref_map_data = ref_map.data

    # Save original parameters of query_map: {origin, step, cell_angles, rotation, symmetries, name}
    grid_query_map_data = query_map.data
    query_map_origin = grid_query_map_data.origin
    query_map_step = grid_query_map_data.step
    query_map_cell_angles = grid_query_map_data.cell_angles
    query_map_rotation = grid_query_map_data.rotation
    query_map_symmetries = grid_query_map_data.symmetries
    query_map_name = grid_query_map_data.name

    grid_ref_map = ref_map.full_matrix().T
    grid_query_map = query_map.full_matrix().T

    vol1 = np.ascontiguousarray(grid_ref_map)
    vol2 = np.ascontiguousarray(grid_query_map)

    # Handle the case where vol1 is 4D:
    if (vol1.ndim == 4) and (vol1.shape[-1] == 1):
        vol1 = np.squeeze(vol1)
    elif vol1.ndim != 3:
        raise UserError("Volumes must be three-dimensional or fourdimensional with singleton first dimension ")

    # Handle the case where vol2 is 4D:
    if (vol2.ndim == 4) and (vol2.shape[-1] == 1):
        vol2 = np.squeeze(vol2)
    elif vol2.ndim != 3:
        raise UserError("Volumes must be three-dimensional or fourdimensional with singleton first dimension ")

    if not ((vol1.shape[1] == vol1.shape[0]) and (vol1.shape[2] == vol1.shape[0])
            and (vol1.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    if not ((vol1.shape[1] == vol1.shape[0]) and (vol1.shape[2] == vol1.shape[0])
            and (vol1.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    if vol1.shape[0] != vol2.shape[0]:
        raise UserError("Input volumes must be of same dimensions")

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = projections
    opt.downsample = downsample

    # Run:
    bestR, bestdx, reflect, vol2aligned, bestcorr = align_volumes_3d.align_volumes(vol1, vol2, opt)

    # Hide the display of query_map (vol2) prior to the alignment:
    query_map.show(show=False)

    # Create GridData object with aligned vol2 but with the original query_map parameters:
    aligned_map_grid_data = arraygrid.ArrayGridData(vol2aligned.T, origin=query_map_origin, step=query_map_step,
                                                    cell_angles=query_map_cell_angles, rotation=query_map_rotation,
                                                    symmetries=query_map_symmetries, name=query_map_name)

    # Replace the data in the original query_map:
    query_map.replace_data(aligned_map_grid_data)

    # fitmap query_map inMap ref_map:
    log.info("Using fitmap to perform final refinement!")
    fitcmd.fit_map_in_map(query_map, ref_map, metric='correlation', envelope=True, zeros=False, shift=True, rotate=True,
                          move_whole_molecules=True, map_atoms=None, max_steps=2000, grid_step_min=0.01, grid_step_max=0.5)

    # Show the query_map (aligned):
    query_map.show(show=True)

    # ******************************************************************************************************************
    # Old code:
    # ******************************************************************************************************************

    # # Set query_map voxel size to ref_map voxel size:
    # ref_map_data = ref_map.data
    # query_map_data = query_map.data
    # ref_map_vsize = ref_map_data.step
    # query_map_vsize = query_map_data.step
    # log.info("ref_map voxel size = " + str(ref_map_vsize))
    # log.info("query_map voxel size = " + str(query_map_vsize))
    # if ref_map_vsize != query_map_vsize:
    #     query_map_data.set_step(ref_map_vsize)
    #     log.info("Updated query_map voxel size to " + str(query_map_data.step))

    # # **************************************************************
    # # CODE BELOW SAVES THE ALIGNED MAP - CAN BE DELETED AFTER TESTS
    # # **************************************************************
    # # Save:
    # # Copy vol2 to save header

    # # PROBLEM - args.vol2 here is the file not the data itself:
    # shutil.copyfile(args.vol2, args.output_vol)

    # # SOLUTION - save first the old vol2:
    # mrc.save(vol2, filename)
    # # Change and save:
    # mrc_fh = mrcfile.open("pre_change_vol2", mode='r+')
    # mrc_fh.set_data(vol2aligned.astype('float32').T)
    # mrc_fh.set_volume()
    # mrc_fh.update_header_stats()
    # mrc_fh.close()
