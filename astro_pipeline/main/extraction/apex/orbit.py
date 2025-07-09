# ---- Match objects with catalog ---------------------------------------------

catalogs = CatalogExtensionPoint('Object Catalogs', Catalog)

# Default catalog
default_catalog = Option(
    'default_catalog', 'USNO-A2.0',
    'Global default catalog ID for all queries', enum=catalogs)




# noinspection PyShadowingBuiltins
def match_objects(objs, cats=None, epoch=None, site=None, filter=None,
                  **keywords):
    """
    Match one or more objects with the specified catalog(s)

    :Parameters:
        - objs   - an instance (or list of instances) of apex.Object to be
                   matched
        - cats   - catalog ID (or a list of IDs) to query; by default, the
                   catalog(s) listed in the default_catalog variable are
                   queried
        - epoch  - optional epoch of catalog query; if specified, should be a
                   datetime.datetime instance (e.g. mid-exposure time)
        - site   - optional site definition: either an IAU observatory code
                   (see apex.sitedef) or a triple (lat, lon, alt) of latitude
                   and longitude (in degrees +North,East) and altitude above
                   MSL (in meters)
        - filter - optional optical filter name; used to compute the object's
                   magnitude in the current instrumental system
        - silent - if True, do not clobber standard output with any messages
                   (useful e.g. for multiple sequential queries); default:
                   False

    :Keywords:
        All other keyword arguments are passed to the underlying query
        functions as their optional parameters

    :Returns:
        A list of the same length as the input object list; each element
        corresponding to an object in "objs" contains either an instance of
        CatalogObject if match was successful or None otherwise; please note
        that the function always returns a list, even for a single input object
        (in the latter case it is a 1-element list)
    """
    # Convert input to list
    if not isinstance(objs, tuple) and not isinstance(objs, list):
        objs = [objs]

    # Check catalog list
    if cats is None:
        cats = default_catalog.value
    if not (isinstance(cats, list) or isinstance(cats, tuple)):
        # Turn a single item to a sequence
        cats = [cats]
    for cat in cats:
        if cat not in catalogs.plugins:
            raise ValueError('Unknown catalog: "{}"'.format(cat))

        check_dynamic(cat, epoch, site, keywords)

    try:
        silent = keywords['silent']
    except KeyError:
        silent = False

    # Scan all selected catalogs
    matches = [None]*len(objs)
    for cat in cats:
        # Indices of unidentified objects
        inds = [i for i, obj in enumerate(matches) if obj is None]
        n = len(inds)
        if not n:
            break

        # Perform match
        catalog = catalogs.plugins[cat]
        if not silent:
            # noinspection PyTypeChecker
            logger.info(
                'Looking for {:d} object(s) in {}'.format(n, catalog.descr))
        new_matches = catalog.match([objs[i] for i in inds], **keywords)
        if not silent:
            # noinspection PyTypeChecker
            logger.info(
                '  {:d} object(s) identified'
                .format(n - new_matches.count(None)))

        # Post-process query result
        normalize_objects([obj for obj in new_matches if obj is not None],
                          catalog, epoch, filter, **keywords)

        # Assign new matches
        for i, obj in zip(inds, new_matches):
            matches[i] = obj

    return matches
