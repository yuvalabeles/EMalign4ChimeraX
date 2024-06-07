from chimerax.map_fit import fitcmd
from chimerax.map_data import arraygrid
import numpy as np
from chimerax.core.errors import UserError
from . import align_volumes_3d


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


def emalign(session, ref_map, query_map, downsample=64, projections=30, show_log=True, show_param=True):
    log = session.logger

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
    bestR, bestdx, reflect, vol2aligned, bestcorr = align_volumes_3d.align_volumes(vol1, vol2, opt=opt, show_log=show_log,
                                                                                   show_param=show_param, session=session)

    # Hide the display of query_map (vol2) prior to the alignment:
    query_map.show(show=False)

    # Create GridData object with aligned vol2 but with the original query_map parameters:
    aligned_map_grid_data = arraygrid.ArrayGridData(vol2aligned.T, origin=query_map_origin, step=query_map_step,
                                                    cell_angles=query_map_cell_angles, rotation=query_map_rotation,
                                                    symmetries=query_map_symmetries, name=query_map_name)

    # Replace the data in the original query_map:
    query_map.replace_data(aligned_map_grid_data)

    # fitmap query_map inMap ref_map:
    if show_log:
        log.info("---> Using fitmap to perform final refinement")
    fitcmd.fit_map_in_map(query_map, ref_map, metric='correlation', envelope=True, zeros=False, shift=True, rotate=True,
                          move_whole_molecules=True, map_atoms=None, max_steps=2000, grid_step_min=0.01, grid_step_max=0.5)

    # Show the query_map (aligned):
    query_map.show(show=True)

    if show_log:
        log.info("* Alignment completed *")
