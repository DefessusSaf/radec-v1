#!/usr/bin/env python
#
# ---------------------------------------------------------------------------------------------------------------------
# Project:     Apex
# Name:        scripts/apex_geo.py
# Purpose:     Apex automatic space object observation processing pipeline
#
# Created:     2005-06-25
# Copyright:   (c) 2004-2024 Vladimir Kouprianov (vkoupr@unc.edu)
# ---------------------------------------------------------------------------------------------------------------------
"""
apex_geo -- Apex automatic space object observation processing pipeline

Usage:
    apex_geo.py [<filename>...] [@<listfile>...]
      [<package>.<option>=<value> ...]

<filename> is the name of the image file, in any supported format, to process.
More than one filename may be specified, and each may include wildcards. List
file names are preceded by the "@" sign. If no filenames are supplied on the
command line, the script will process the whole current working directory,
excluding calibration frames.

Defaults for any option stored in the apex.conf file may be overridden without
affecting the master configuration file by explicitly specifying them on the
command line, like

    python apex_geo.py 1394*.fit 2004MN4*.fit extraction.threshold=5

Option values containing spaces should be enclosed in double quotes:

    python apex_geo.py some_file.fit io.default_format="my format"

For each space object within the frame to be identified and reported, the
script requires a pair of approximate X and Y pixel coordinates of the GEO,
stored in the FITS header. The corresponding FITS keywords are expected to be
"OBJX" and "OBJY" for the target object, "OBJ01X" and "OBJ01Y" for the possible
additional space objects appearing within the frame, "OBJ02X" and "OBJ02Y" for
the next one, and so forth.

In case when reference stars are too faint for the automatic extraction
mechanism to identify them, one may specify XY positions of reference stars
explicitly, using "STAR001X"/"STAR001Y" etc. keyword pairs. When the script
finds these keywords in the FITS header, it fully disables automatic star
extraction pipeline, which also reduces the processing time.

After successful identification of the targets, the script reports their RA and
Dec positions and magnitudes, along with the frame name and mid-exposure time.
The actual report file format depends upon the value of the
[apex.extra.GEO.report].format option.

Script-specific options:
  disable_calib = 0 | 1
    - turn off automatic image calibration
  disable_measurement = 0 | 1
    - disable PSF fitting stage; use barycenter/manual XY positions of objects
  ignore_refstars = 0 | 1
    - ignore explicit XY positions of reference stars in the image header
  overwrite_frame = 0 | 1
    - overwrite the original frame file after processing
  target_match_tol = <non-negative number>
    - manual target XY position accuracy, px
  trail_len_tol = <non-negative number>
    - star trail length tolerance, px (0 to disable trail length check)
  trail_width_tol = <non-negative number>
    - star trail width tolerance factor (0 to disable trail width check)
  auto_postfilter = 0 | 1
    - automatically choose between trail_cluster and cluster postfilter based
      on the expected star trail length; overrides
      apex.extraction.postfilter_chain
"""

# Import all required modules
# Note. Any of the Apex library modules should be imported prior to non-builtin
#       Python modules in frozen mode since they all are in apex.lib and become
#       accessible only after apex/__init__.py is loaded.
import apex.io
import apex.conf
import apex.sitedef
import apex.util.automation.calibration as calib_util
import apex.extraction.filtering
import apex.measurement.psf_fitting
import apex.measurement.rejection
import apex.measurement.util
import apex.measurement.aperture
import apex.identification.main
import apex.identification.util
import apex.astrometry.reduction
import apex.astrometry.util
import apex.photometry.aperture
import apex.photometry.differential
import apex.parallel
import apex.extra.GEO
from apex.extra.GEO.report import report_measurements, adjust_detection_attrs
from apex.extra.GEO.util.refstars import filter_refstars
from apex.logging import *
import apex.util.report.table
from apex.util.file import get_cmdline_filelist

import sys
import time
import os.path
import numpy as np
import scipy.ndimage


# Script-specific options
disable_calib = apex.conf.Option('disable_calib', False, 'Turn off automatic image calibration')
overwrite_frame = apex.conf.Option('overwrite_frame', False, 'Overwrite the original frame file after processing')
target_match_tol = apex.conf.Option(
    'target_match_tol', 5.0, 'Manual target XY position accuracy, px', constraint='target_match_tol >= 0')
trail_len_tol = apex.conf.Option(
    'trail_len_tol', 5.0, 'Star trail length tolerance, px (0 to disable trail length check)',
    constraint='trail_len_tol >= 0')
trail_width_tol = apex.conf.Option(
    'trail_width_tol', 3.0, 'Star trail width tolerance factor (0 to disable trail width check)',
    constraint='trail_width_tol >= 0')
auto_postfilter = apex.conf.Option(
    'auto_postfilter', True,
    'Automatically choose between trail_cluster and cluster postfilter based on the expected star trail length')
disable_measurement = apex.conf.Option(
    'disable_measurement', False, 'Disable PSF fitting stage; use barycenter/manual XY positions of objects')
ignore_refstars = apex.conf.Option(
    'ignore_refstars', False, 'Ignore explicit XY positions of reference stars in the image header')


def identified_objects(img):
    return [obj for obj in img.objects if hasattr(obj, 'match') or hasattr(obj, 'phot_match')]


def unidentified_objects(img):
    return [obj for obj in img.objects if not hasattr(obj, 'match') and not hasattr(obj, 'phot_match')]


# Distortion map
distmap = None


# Apply displacement vector to each object within the image by interpolation of
# the distortion matrix
# noinspection DuplicatedCode
def apply_distmap(img, objects=None):
    if objects is None:
        objects = img.objects
    n, m = distmap[0].shape
    logger.info('\nApplying a {:d} x {:d} distortion map to {:d} object(s)'.format(m, n, len(objects)))
    kx, ky = m/img.width, n/img.height

    for obj in objects:
        # Convert XY image coordinates to cell number and interpolate the distortion map
        dx, dy = [scipy.ndimage.map_coordinates(
            m, ([obj.Y*ky - 0.5], [obj.X*kx - 0.5]), mode='nearest', prefilter=False)[0] for m in distmap]
        obj.X -= dx
        obj.Y -= dy


def int_or_str(obj_id):
    # noinspection PyBroadException
    try:
        return int(obj_id)
    except Exception:
        return str(obj_id).strip()


# Main processing function
def _process_image(filename, darks, flats, logfile):
    # Load the image
    logger.info('\n\n' + '-'*80 + 'Processing image "{}"\n'.format(filename))
    img = apex.io.imread(filename)
    if not hasattr(img, 'target_pos') or not len(img.target_pos):
        logger.warning(
            '\nNo target positions found in the FITS header.\nSpace object identification and reporting disabled')
    if hasattr(img, 'refstar_pos') and len(img.refstar_pos):
        logger.info('Automatic object extraction disabled')

    # If no site info found in the image header, use the default longitude from apex.conf -- required for RA <-> HA
    # conversion
    if not hasattr(img, 'sitelon'):
        img.sitelon = apex.sitedef.longitude.value

    # Save the image copy for future reference
    orig_img = img.copy()

    # Compute the expected star trail length, rotation, and width, in pixels
    expected_trail_len, expected_trail_rot, expected_trail_width = apex.extraction.filtering.get_trail_shape(img)

    # Calibrate the image; do not subtract sky
    if not disable_calib.value:
        logger.info('\n\n' + '-'*80 + '\nImage calibration')
        calib_util.correct_all(img, darks, flats)

    if hasattr(img, 'refstar_pos') and img.refstar_pos and not ignore_refstars.value:
        # Forced reference stars found; simulate an apex.Object structure for each one and for the targets
        k = 2*np.sqrt(2*np.log(2))
        roi_a = expected_trail_len/k
        roi_b = expected_trail_width/k
        aperture_factor = apex.measurement.aperture.aperture_factor.value or \
            apex.measurement.aperture.find_optimal_aperture_factor(img)

        def create_refstar(_x, _y):
            star = apex.Object()
            star.cent_X, star.cent_Y = _x, _y
            star.roi_a, star.roi_b = roi_a, roi_b
            star.roi_rot = expected_trail_rot
            star.pixels_X, star.pixels_Y, star.I, star.annulus_X, star.annulus_Y, star.annulus_I = \
                apex.measurement.aperture.get_aperture(
                    img, _x, _y, expected_trail_len, expected_trail_width, expected_trail_rot, True, 'trailed',
                    aperture_factor=aperture_factor)[3:]
            return star
        img.objects = [create_refstar(x, y) for (x, y), _ in img.refstar_pos]
    else:
        # Detect stars and other objects using the default object extractor
        logger.info('\n\n' + '-'*80 + '\nObject detection\n')
        apex.util.report.print_extraction_options()

        # Override postfilter_chain if auto_postfilter selected: choose [cluster] if ratio of expected star trail length
        # and width is below trail_threshold and [trail_cluster] otherwise
        kw = {}
        if auto_postfilter.value:
            if expected_trail_len/expected_trail_width > apex.measurement.psf_fitting.trail_threshold.value:
                postfilter_chain = ['trail_cluster']
            else:
                postfilter_chain = ['cluster']
            if postfilter_chain != apex.extraction.main.postfilter_chain.value:
                logger.info('\nForced post-filter chain: {}'.format(postfilter_chain))
                kw['custom_postfilters'] = postfilter_chain

        ndet = apex.extraction.detect_objects(img, **kw)
        if ndet:
            logger.info('\n{:d} object(s) detected in the image'.format(ndet))
        else:
            raise TerminatePipeline('No objects could be detected with the current parameters')

    try:
        # Measure positions by fitting profiles
        if not disable_measurement.value:
            logger.info('\n\n' + '-'*80 + '\nPositional measurement\n')
            if hasattr(img, 'refstar_pos') and img.refstar_pos and not ignore_refstars.value:
                # When refstars are explicitly specified, completely disable object rejection before PSF fitting;
                # though, discard objects outside the image boundary after PSF fitting
                apex.measurement.rejection.rejector_pre_sequence.tmpvalue = []
                apex.measurement.rejection.rejector_post_sequence.tmpvalue = ['boundary']
            apex.util.report.print_measurement_options()
            img_objects = list(img.objects)
            ndet = apex.measurement.psf_fitting.measure_objects(img)
            if ndet:
                logger.info('\n{:d} object(s) measured successfully'.format(ndet))
            else:
                logger.warning('\nNo objects could be measured; using isophotal analysis/manual XY positions')
                # Restore the list of detected objects which has been cleared by measure_objects()
                img.objects = img_objects
            del img_objects

        # Run aperture photometry
        logger.info('\n\n' + '-'*80 + '\nAperture photometry\n')
        apex.util.report.print_aperture_photometry_options()
        apex.photometry.aperture.aperture_photometry(img)

        # Extract all targets
        targets = []
        if hasattr(img, 'target_pos') and len(img.target_pos):
            # Identify all items in the list of detected objects using the nearest neighbor match algorithm, with
            # the given position tolerance
            if img.objects:
                matches = apex.identification.util.neighbor_match(
                    [(obj.X, obj.Y) for obj in img.objects],
                    [item[1] for item in img.target_pos],
                    target_match_tol.value).tolist()
            else:
                matches = []
            # Find and identify targets by proximity in the XY space to the target position, read out manually from
            # the image before processing
            for target_num, (name, xy, trail) in enumerate(img.target_pos):
                if target_num in matches:
                    obj = img.objects[matches.index(target_num)]
                    logger.info(
                        '\nTarget {} ({:.1f},{:.1f}) found at ({:.1f},{:.1f})'.format(name, xy[0], xy[1], obj.X, obj.Y))
                else:
                    # No object found at the specified XY position; force
                    # detection and measurement
                    logger.warning(
                        '\nTarget {} could not be found automatically; forcing detection at the manually specified XY '
                        'location {}'.format(
                            name,
                            '({0[0]:.1f},{0[1]:.1f})-({1[0]:.1f},{1[1]:.1f})'.format(
                                trail[0], trail[1]) if trail else '({0[0]:.1f},{0[1]:.1f})'.format(xy)))
                    obj = apex.measurement.util.force_measurement(img, *(trail if trail else xy))
                    if hasattr(obj, 'roi_a') and hasattr(obj, 'roi_b'):
                        logger.info(
                            'Isophotal analysis produces a {:.1f}x{:.1f} px object at ({:.1f},{:.1f})'.format(
                                2*np.sqrt(2*np.log(2))*obj.roi_a, 2*np.sqrt(2*np.log(2))*obj.roi_b,
                                obj.cent_X, obj.cent_Y))
                        if hasattr(obj, 'psf'):
                            logger.info(
                                'PSF fitting produces a {:.1f}x{:.1f} px object at ({:.1f},{:.1f})'.format(
                                    obj.FWHM_X, obj.FWHM_Y, obj.X, obj.Y))
                            if np.hypot(obj.X - xy[0], obj.Y - xy[1]) > target_match_tol.value:
                                if np.hypot(obj.cent_X - xy[0], obj.cent_Y - xy[1]) > target_match_tol.value:
                                    logger.warning(
                                        'PSF and isophotal centroids too far from the expected position; using '
                                        'manual XY position')
                                    obj.X, obj.Y = xy
                                else:
                                    logger.warning(
                                        'PSF centroid too far from the expected position; using isophotal centroid')
                                    obj.X, obj.Y = obj.cent_X, obj.cent_Y
                        else:
                            logger.error('PSF fitting for target failed')
                            if np.hypot(obj.cent_X - xy[0], obj.cent_Y - xy[1]) > target_match_tol.value:
                                logger.warning(
                                    'Isophotal centroids too far from the expected position; using manual XY position')
                                obj.X, obj.Y = xy
                            else:
                                logger.info('Using isophotal centroid')
                    else:
                        logger.error('Isophotal analysis for target failed; using manual XY position')
                    logger.warning('\nTarget {} measured forcibly; low accuracy is possible'.format(name))

                # Mark each object identified by the "target" flag, remember its manually assigned identifier
                obj.flags.add('target')
                obj.id = name
                targets.append(obj)

            # Remove all targets from the list of measured objects (which will serve as the list of reference stars)
            [img.objects.remove(obj) for obj in targets if obj in img.objects]
        if targets:
            logger.info('\n\n{:d} target(s) found at the specified XY location(s)'.format(len(targets)))
        else:
            logger.warning('\n\nNo targets found')
        ndet = len(img.objects)

        # Leave only good reference stars
        filter_refstars(
            img, expected_trail_len, expected_trail_width, trail_len_tol.value,
            trail_width_tol.value)

        # Perform plate reduction
        logger.info('\n\n' + '-'*80 + '\nReference catalog matching and astrometric reduction\n')
        apex.util.report.print_astrometry_options()

        logger.info('')
        if distmap is not None:
            apply_distmap(img)
        try:
            apex.astrometry.reduction.reduce_plate(img)
        except Exception as e:
            logger.warning('\nAstrometric reduction failed:', e)
        if not img.wcs.reduction_model:
            img.objects += targets
            raise TerminatePipeline('Could not find LSPC solution')
        apex.util.report.print_lspc_report(img)

        # Compute RA/Dec positions of targets
        if targets:
            apex.astrometry.util.calc_coords(img, targets)

        # Perform the final lookup for unidentified objects, if any
        logger.info('\n\n' + '-'*80 + '\nLookup for unidentified objects\n')
        if unidentified_objects(img):
            apex.identification.identify_all(img)
        else:
            logger.info('No unidentified objects left')

        # Return all targets back
        img.objects += targets

        # Perform differential photometric reduction
        if ndet:
            logger.info('\n\n' + '-'*80 + '\nPhotometric reduction\n')
            apex.util.report.print_photometry_options()
            apex.photometry.differential.photometry_solution(img)

            # Remember seeing
            # noinspection PyBroadException
            try:
                from apex.util.seeing import compute_seeing
                img.seeing = compute_seeing(img.objects, img.wcs)
            except Exception:
                pass

    finally:
        # Display image info
        logger.info('\n\n' + '-'*80 + '\nImage information\n')
        apex.util.report.print_image_info(img)

        # Save processing results
        logger.info('\n\n' + '-'*80 + '\nProcessing summary\n')
        apex.util.report.print_processing_summary(img, orig_img)

        if img.objects:
            # Sort detected objects; put space object candidates first
            ras = [s.ra % 24 for s in img.objects if hasattr(s, 'ra')]
            wrap = any(ra >= 12 for ra in ras) and any(ra < 12 for ra in ras)
            min_x = min(s.X for s in img.objects)
            xr = max(s.X for s in img.objects) - min_x
            img.objects.sort(
                key=lambda o: (o.ra % 24 - 24 if wrap and o.ra >= 12 else o.ra % 24) - (xr + 24)*('target' in o.flags)
                if hasattr(o, 'ra') else o.X - min_x + (-xr if 'target' in o.flags else 24))

            # Output catalog of objects (to the log file only)
            print('\n\nCatalog of objects:', file=logfile)
            apex.util.report.table.output_catalog(img.objects, dest=logfile)

            # Report all targets found
            logger.info('\n\n' + '-'*80 + '\nTarget report\n')
            apex.util.report.print_object_list(
                img, lambda o: 'target' in o.flags and hasattr(o, 'ra') and hasattr(o, 'dec'), 'NONE FOUND')
            with apex.parallel.main.pool_lock:
                # Append observations to the appropriate report file(s); allow this only for one subprocess at a time
                try:
                    report_measurements([
                        adjust_detection_attrs(img, obj) for obj in img.objects
                        if 'target' in obj.flags and hasattr(obj, 'ra') and hasattr(obj, 'dec')])
                except Exception as e:
                    logger.exception(
                        '\nCould not create report [{}]\n         '
                        'See catalog above for RA and Dec position'.format(e))

        # Save the image file with all processing result, overwriting the original one
        if overwrite_frame.value:
            apex.io.imwrite(img, img.filename, img.fileformat)

        # Maintain the list of frames where each object is present
        objects = {}
        with apex.parallel.main.pool_lock:
            # noinspection PyBroadException
            try:
                for line in open('frames.txt', 'r').read().splitlines():
                    # noinspection PyBroadException
                    try:
                        obj_id, names = map(str.strip, line.split(':'))
                        if obj_id:
                            objects[int_or_str(obj_id)] = {s.strip() for s in names.split(',')}
                    except Exception:
                        pass
            except Exception:
                pass
            fn = os.path.split(img.filename)[-1]
            obj_ids = {int_or_str(obj.id) for obj in img.objects if 'target' in obj.flags}
            # Identify objects that are already present in frames.txt and claim to belong to the current frame but
            # actually don't
            for obj_id, names in objects.items():
                if fn in names and obj_id not in obj_ids:
                    names.remove(fn)
            # Add the current frame to items for all its detections
            for obj_id in obj_ids:
                if obj_id in objects:
                    objects[obj_id].add(fn)
                else:
                    objects[obj_id] = {fn}
            # Remove empty objects
            for obj_id in list(objects.keys()):
                if not objects[obj_id]:
                    del objects[obj_id]
            # Save sorted objects back to frames.txt
            with open('frames.txt', 'wt') as f:
                for obj_id in sorted(objects):
                    print('{}: {}'.format(obj_id, ', '.join(sorted(objects[obj_id]))), file=f)

    # The end
    logger.info('\n\n' + '-'*80 + '\nImage processing pipeline complete')


def process_image(filename, _, darks, flats):
    starttime = time.time()

    # Initiate logging
    log_filename = os.path.splitext(filename)[0] + '.proclog'
    logfile = start_logging(log_filename, 'Starting Apex automatic space object image processing pipeline')
    logger.info(f'\nApex/GEO package version: {apex.extra.GEO.__version__}')
    apex.util.report.print_module_options('__main__', '\nScript-specific options')

    try:
        # noinspection PyBroadException
        try:
            _process_image(filename, darks, flats, logfile)
        except TerminatePipeline as e:
            # Explicit pipeline termination
            logger.error('\n\nPremature pipeline termination:\n{}'.format(e))
        except Exception:
            # Unexpected exception caught
            logger.critical('\n\nAbnormal pipeline termination. Traceback follows:\n', exc_info=True)
    finally:
        # Report the processing time
        logger.info('\n\nProcessing time: {:.0f}m {:g}s'.format(*divmod(time.time() - starttime, 60)))

        # Stop logging for the current file
        stop_logging(log_filename)


# noinspection DuplicatedCode
def main():
    global distmap

    # Remember the starting time of the script
    script_starttime = time.time()

    # Obtain the list of files to process from the command line
    filenames = get_cmdline_filelist()
    if not filenames:
        print('\nNo files to process', file=sys.stderr)
        sys.exit(2)

    # Load all necessary calibration frames
    if disable_calib.value:
        darks, flats = {}, {}
    else:
        darks = calib_util.load_darks(filenames)
        flats = calib_util.load_flats(filenames)

    # Read the optional distortion map; distmap[0] and distmap[1] are matrices of X and Y displacements, respectively
    # noinspection PyBroadException
    try:
        distmap = [scipy.ndimage.spline_filter(m, 3)
                   for m in np.swapaxes(np.asarray(
                       [zip(*[[float(s) for s in item.split(';')] for item in line.split()])
                        for line in open('distmap.dat', 'r').read().splitlines()]), 0, 1)]
    except Exception:
        pass

    # Process all files
    apex.parallel.parallel_loop(process_image, filenames, args=(darks, flats), backend='mp')

    # Report the full script execution time
    logger.info('\nTotal time elapsed: {:.0f}m {:g}s'.format(*divmod(time.time() - script_starttime, 60)))


if __name__ == '__main__':
    main()
