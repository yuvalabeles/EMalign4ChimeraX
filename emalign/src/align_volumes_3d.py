#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:06:16 2021

@author: yaelharpaz1
"""
import logging
# import random

import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation
from .common_finufft import cryo_downsample
from .cryo_project_itay_finufft import cryo_project
from .genRotationsGrid import genRotationsGrid
from .align_projection_2d import align_projection
from .fastrotate3d import fastrotate3d
from . import fastrotate3d
from .register_translations_3d import register_translations_3d
from .reshift_vol import reshift_vol
from . import reshift_vol

# np.random.seed(10)


def fast_alignment_3d(vol1, vol2, Nprojs=30, verbose=0, log=None, show_log=True):
    """
    This function does the work for align_volumes.

    Input:
    vol1 - 3D reference volume that vol2 should be aligned accordingly.
    vol2 - 3D volume to be aligned.
    verbose - set verbose to nonzero for verbose printouts (default is zero).
    Nprojs - number of reference projections for the alignment.

    Output:
    Rest - the estimated rotation between vol_2 and vol_1 without reflection.
    Rest_J - the estimated rotation between vol_2 and vol_1 with reflection.
    """
    # np.random.seed(1)

    logger = logging.getLogger()
    if verbose == 0:
        logger.disabled = True

    # Generate reference projections from vol2:
    if show_log:
        log.info(f'---> Generating {Nprojs} reference projections')
    Rots = genRotationsGrid(75)
    sz_Rots = np.size(Rots, 2)
    rots_z = np.random.randint(sz_Rots, size=Nprojs)

    # rots_z = np.array([10398, 3521, 8847, 2453, 2060, 12926, 5752, 2003, 13819, 9481, 2796, 13563, 6517, 12998, 7628,
    #                    806, 3366, 8271, 11801, 1831, 9333, 1372, 4976, 9121, 312])
    # log.info(f'---> random ints from \n{sz_Rots}\n to \n{rots_z}\n')

    R_ref = Rots[:, :, rots_z]  # size (3,3,N_projs)
    # R_ref = mat_to_npy('R_ref_for_fastAlignment3D')
    ref_projs = cryo_project(vol2, R_ref)
    ref_projs = np.transpose(ref_projs, (1, 0, 2))
    R_ref = np.transpose(R_ref, (1, 0, 2))  # the true rotations.
    # log.info(f'ref_projs:\n{ref_projs}')
    # log.info(f'R_ref:\n{R_ref}')

    # Align reference projections to vol1:
    class Struct:
        """
        Used to pass optimal paramters to the alignment function
        """
        pass

    opt = Struct()
    opt.Nprojs = Nprojs
    # opt.Rots = Rots

    print_to_log(log, "---> Aligning reference projections of query map to reference map", show_log)

    R_tild = align_projection(ref_projs, vol1, verbose, opt)  # size (3,3,N_projs).

    # Synchronization:
    # A synchronization algorithm is used in order to revel the symmetry elements of the reference projections.
    # Estimate X with or without reflection:
    R_tild = R_tild[0]
    X_mat = np.zeros((3, 3, Nprojs))
    X_mat_J = np.zeros((3, 3, Nprojs))
    J3 = np.diag([1, 1, -1])
    for i in range(Nprojs):
        X_mat[:, :, i] = R_ref[:, :, i] @ R_tild[:, :, i].T
        X_mat_J[:, :, i] = R_ref[:, :, i] @ (J3 @ R_tild[:, :, i] @ J3).T

    # Construct the synchronization matrix with and without reflection:
    X_ij = np.zeros((3 * Nprojs, 3 * Nprojs))
    X_ij_J = np.zeros((3 * Nprojs, 3 * Nprojs))
    for i in range(Nprojs):
        for j in range(i + 1, Nprojs):
            X_ij[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat[:, :, i].T @ X_mat[:, :, j]
            X_ij_J[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat_J[:, :, i].T @ X_mat_J[:, :, j]

    # Enforce symmetry:
    X_ij = X_ij + X_ij.T
    X_ij = X_ij + np.eye(np.size(X_ij, 0))
    X_ij_J = X_ij_J + X_ij_J.T
    X_ij_J = X_ij_J + np.eye(np.size(X_ij_J, 0))

    # Define v=[g_1.',..., g_N_projs.'].' (v is of size 3*N_projx3), then X_ij=v*v.', and Xij*v=N_projs*v.
    # Thus, v is an eigenvector of Xij. The matrix Xij should be of rank 3.
    # Find the top 3 eigenvectors:

    # Without reflection:
    s, U = la.eigh(X_ij)  # s = np.diag(s);
    ii = np.argsort(s, axis=0)[::-1]  # s = np.sort(s,axis=0)[::-1]
    U = U[:, ii]
    V = U[:, 0:3]
    # With reflection:
    sJ, UJ = la.eigh(X_ij_J)  # sJ = np.diag(sJ);
    iiJ = np.argsort(sJ, axis=0)[::-1]  # sJ = np.sort(sJ,axis=0)[::-1];
    UJ = UJ[:, iiJ]
    VJ = UJ[:, 0:3]

    # Estimating G:
    # Estimate the group elemnts for each reference image.
    # G denotes the estimated group without reflection, and G_J with reflection. This estimation is being done from
    # the eigenvector v by using a rounding algorithm over SO(3) for each 3x3 block of v.
    G = np.zeros((Nprojs, 3, 3))
    G_J = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        B = V[3 * i:3 * (i + 1), :]
        u_tmp, _, v_tmp = la.svd(B)
        B_round = la.det(u_tmp @ v_tmp) * (u_tmp @ v_tmp)
        G[i, :, :] = B_round.T
        # reflected case:
        BJ = VJ[3 * i:3 * (i + 1), :]
        uJ_tmp, _, vJ_tmp = la.svd(BJ)
        BJ_round = la.det(uJ_tmp @ vJ_tmp) * (uJ_tmp @ vJ_tmp)
        G_J[i, :, :] = BJ_round.T

    # Set the global rotation to be an element from the symmetry group:
    # The global rotation from the synchronization can be any rotation matrix from SO(3).
    # So, in order to get the estimated symmetry elements to be from the symmetry group we set the global rotation
    # to be also an element from the symmetry group.
    O1 = G[0, :, :].T
    O1_J = G_J[0, :, :].T
    G_est = np.zeros((Nprojs, 3, 3))
    G_J_est = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        G_est[i, :, :] = O1 @ G[i, :, :]
        G_J_est[i, :, :] = O1_J @ G_J[i, :, :]

    # Estimating the rotation:
    # Estimate the two candidate orthogonal transformations.
    for i in range(Nprojs):
        X_mat[:, :, i] = X_mat[:, :, i] @ G_est[i, :, :].T
        X_mat_J[:, :, i] = X_mat_J[:, :, i] @ G_J_est[i, :, :].T
    X = np.mean(X_mat, axis=2)
    X_J = np.mean(X_mat_J, axis=2)
    # Without reflection:
    R = X
    U, _, V = la.svd(R)  # Project R to the nearest rotation.
    R_est = U @ V
    assert la.det(R_est) > 0
    R_est = R_est[:, [1, 0, 2]][[1, 0, 2]]
    R_est = R_est.T
    # Reflected case:
    R_J = X_J
    U, _, V = la.svd(R_J)  # Project R to the nearest rotation.
    R_est_J = U @ V
    assert la.det(R_est_J) > 0
    R_est_J = R_est_J[:, [1, 0, 2]][[1, 0, 2]]
    R_est_J = R_est_J.T

    logging.shutdown()

    return R_est, R_est_J


# %%
def evalO(X, R_true, R_est, G):
    psi = X[0]
    theta = X[1]
    phi = X[2]
    O_mat = Rotation.as_matrix(Rotation.from_euler('xyz', [psi, theta, phi], degrees=False))
    n = np.size(G, 0)
    dist = np.zeros((1, n))
    for i in range(n):
        g = G[i, :, :]
        dist[0, i] = la.norm(R_true - O_mat @ g @ O_mat.T @ R_est, 'fro')
    err = np.min(dist)
    return err


# %%
def align_volumes(vol1, vol2, verbose=0, opt=None, show_log=True, session=None):
    """
    This function aligns vol2 according to vol1.
    Aligning vol2 to vol1 by finding the relative rotation, translation and reflection between vol1 and vol2,
    such that vol2 is best aligned with vol1.
    How to align the two volumes:
        The user should align vol2 according to vol1 using the parameters bestR, bestdx and reflect.
        If reflect=0 then there is no reflection between the volumes.
        In that case the user should first rotate vol2 by bestR and then reshift by bestdx.
        If reflect=1, then there is a reflection between the volumes.
        In that case the user should first reflect vol2 about the z axis using the flip function,
        then rotate the volume by bestR and finally reshift by bestdx.
    Input:
        vol1- 3D reference volume that vol2 should be aligned accordingly.
        vol2- 3D volume to be aligned.
        verbose - Set verbose to nonzero for verbose printouts (default is zero).
    Output:
        bestR - the estimated rotation between vol2 and vol1, such that bestR*vol2 will align vol2 to vol1.
        bestdx - size=3x1. the estimated translation between vol2 and vol1.
        reflect - indicator for reflection.
                  If reflect=1 then there is a reflection between vol1 and vol2, else reflect=0.
                  In order to align the volumes in the case of reflect=1, the user should first reflect vol2 about
                  the z axis, and then rotate by bestR.
        vol2aligned - vol2 after applying the est. transformation, so it's best aligned with vol1 (after optimization).
        bestcorr - the coorelation between vol1 and vol2aligned.
    Options:
        opt.downsample - Downsample the volume to this size (in pixels) for faster alignment. Default is 64.
                         Use larger value if alignment fails.
        opt.Nprojs - Number of projections to use for the alignment. Defult is 30.
    """

    logger = logging.getLogger()

    if session is not None:
        log = session.logger
    else:
        log = None

    if verbose is False:
        logger.disabled = True
    else:
        logger.disabled = False

    class Struct:
        pass

    # Check options:
    if opt is None:
        opt = Struct()
        opt.no_refine = True

    if not hasattr(opt, 'no_refine'):
        opt.no_refine = False

    # Check options:
    if hasattr(opt, 'Nprojs'):
        Nprojs = opt.Nprojs
    else:
        Nprojs = 30

    if hasattr(opt, 'downsample'):
        downsample = opt.downsample
    else:
        downsample = None

    if hasattr(opt, 'align'):
        align_in_place = opt.align[0]
    else:
        align_in_place = True

    # Validate input:
    # Input volumes must be 3-dimensional, where all dimensions must be equal.
    # This restriction can be removed, but then, the calculation of nr (radial resolution in the Fourier domain)
    # should be adjusted accordingly. Both vol_1 and vol_2 must have the same dimensions.

    n_1 = np.shape(vol1)
    n_2 = np.shape(vol2)

    assert np.size(n_1) == 3 and np.size(n_2) == 3, "Inputs must be 3D"
    assert n_1[0] == n_1[1] == n_1[2], "All dimensions of input volumes must be equal"
    assert n_2[0] == n_2[1] == n_2[2], "All dimensions of input volumes must be equal"

    # Downsampling speeds up calculation, and does not seem to degrade accuracy:
    if n_1[0] == n_2[0]:
        n = n_1[0]
        n_ds = downsample
        if downsample is None:
            assert n <= 256, "Input volume is more than 256 pixels, must downsample"
            print_to_log(log, f"---> Downsampling volumes from {n} to {n_ds} pixels", show_log=show_log)
            vol1_ds = cryo_downsample(vol1, (n_ds, n_ds, n_ds))
            vol2_ds = cryo_downsample(vol2, (n_ds, n_ds, n_ds))
        else:
            assert downsample <= n, "Downsample must be less than input volume size"
            print_to_log(log, f"---> Downsampling volumes from {n} to {n_ds} pixels", show_log=show_log)
            vol1_ds = cryo_downsample(vol1, (n_ds, n_ds, n_ds))
            vol2_ds = cryo_downsample(vol2, (n_ds, n_ds, n_ds))
        print_to_log(log, f"---> Referencing volumes sized ({n_ds},{n_ds},{n_ds})", show_log=show_log)
    else:
        assert n_1[0] == n_2[0], "ERROR"
        vol1_ds = vol1
        vol2_ds = vol2

    # Aligning the volumes:
    R_est, R_est_J = fast_alignment_3d(vol1_ds.copy(), vol2_ds.copy(), Nprojs, verbose, log, show_log)

    logger.debug("R_est=\n%s", str(R_est))
    logger.debug("R_est_J=\n%s", str(R_est_J))

    vol2_aligned_ds = fastrotate3d.fastrotate3d(vol2_ds, R_est)  # rotate the original vol_2 back
    vol2_aligned_J_ds = fastrotate3d.fastrotate3d(vol2_ds, R_est_J)

    vol2_aligned_J_ds = np.flip(vol2_aligned_J_ds, axis=2)
    estdx_ds = register_translations_3d(vol1_ds, vol2_aligned_ds)
    estdx_J_ds = register_translations_3d(vol1_ds, vol2_aligned_J_ds)
    logger.debug("estdx_ds=%s", str(estdx_ds))
    logger.debug("estdx_J_ds=%s", str(estdx_J_ds))

    if np.size(estdx_ds) != 3 or np.size(estdx_J_ds) != 3:
        raise Warning("***** Translation estimation failed *****")

    # Prepare FFTW data to avoid unnecessary calaculations:
    vol2_aligned_ds = reshift_vol.reshift_vol_int(vol2_aligned_ds, estdx_ds)
    vol2_aligned_J_ds = reshift_vol.reshift_vol_int(vol2_aligned_J_ds, estdx_J_ds)

    no1 = np.mean(np.corrcoef(vol1_ds.ravel(), vol2_aligned_ds.ravel(), rowvar=False)[0, 1:])
    no2 = np.mean(np.corrcoef(vol1_ds.ravel(), vol2_aligned_J_ds.ravel(), rowvar=False)[0, 1:])
    logger.debug("no1=%f", no1)
    logger.debug("no2=%f", no2)

    reflect = 0
    corr_v = no1
    if no2 > no1:
        J3 = np.diag([1, 1, -1])
        corr_v = no2
        R_est = R_est_J
        R_est = J3 @ R_est @ J3
        estdx_ds = estdx_J_ds
        # vol2_ds = np.flip(vol2_ds, axis=2)
        vol2 = np.flip(vol2, axis=2)
        reflect = 1
        print_to_log(log, "---> Input volumes are reflected w.r.t each other", show_log=show_log)

    print_to_log(log, f"---> Correlation between downsampled aligned volumes is {corr_v:.4f}", show_log=show_log)

    logger.debug("R_est=\n%s", str(R_est))
    logger.debug("estdx_ds=%s", str(estdx_ds))
    logger.debug("reflect=%d", reflect)

    # Refinement option is disabled, therefore:
    bestR = R_est

    logger.debug("bestR=\n%s", str(bestR))

    print_to_log(log, "---> Done aligning downsampled volumes\n"
                      "---> Applying estimated rotation to volumes", show_log=show_log)

    vol2aligned = fastrotate3d.fastrotate3d(vol2, bestR)

    print_to_log(log, "---> Estimating shift for volumes", show_log=show_log)

    bestdx = register_translations_3d(vol1, vol2aligned)
    logger.debug("bestdx=%s", str(bestdx))

    if not align_in_place:
        translation_msg = f"---> Translations before pixel adjustment: [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]"
        print_to_log(log, translation_msg, show_log=show_log)

    if align_in_place:
        print_to_log(log, "---> Translating volumes\n", show_log=show_log)
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            vol2aligned = reshift_vol.reshift_vol_int(vol2aligned, bestdx)
        else:
            vol2aligned = reshift_vol.reshift_vol(vol2aligned, bestdx)
    else:
        vol2aligned = vol2

    logging.shutdown()

    vol2aligned = vol2aligned.astype(np.float32)

    return bestR, bestdx, reflect, vol2aligned


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Helper Functions: ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#
# def print_param(log, bestR, bestdx, show_param):
#     if show_param:
#         log.info('Estimated Rotation:')
#         log.info(f'[[{bestR[0, 0]:.3f} {bestR[0, 1]:.3f} {bestR[0, 2]:.3f}],')
#         log.info(f'[{bestR[1, 0]:.3f} {bestR[1, 1]:.3f} {bestR[1, 2]:.3f}]')
#         log.info(f'[{bestR[2, 0]:.3f} {bestR[2, 1]:.3f} {bestR[2, 2]:.3f}]]')
#         log.info(f'Estimated Translations: \n[{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]\n')


def print_to_log(log, msg, show_log=True):
    if show_log:
        log.info(msg)
