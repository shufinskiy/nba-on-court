from typing import Callable, Sequence

import numpy as np
import pandas as pd


BINWIDTHS = np.array([1.5, 1.5])


def round_any(x: float, accuracy: float, f: Callable=round):
    """

    Args:
        x:
        accuracy:
        f:

    Returns:

    """
    return f(x / accuracy) * accuracy


def hex_bounds(x: Sequence, binwidth: float):
    """

    Args:
        x:
        binwidth:

    Returns:

    """
    return np.array([round_any(np.min(x), binwidth, np.floor) - 1e-6,
                     round_any(np.max(x), binwidth, np.ceil) + 1e-6])


def hexbin(x, y, xbins, xbnds, ybnds, shape):
    """

    Args:
        x:
        y:
        xbins:
        xbnds:
        ybnds:
        shape:

    Returns:

    """
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(x, pd.Series):
        y = y.to_numpy()
    n = len(x)
    jmax = np.floor(xbins + 1.5001)
    c1 = 2 * np.floor((xbins * shape) / np.sqrt(3) + 1.5001)
    imax = np.trunc((jmax * c1 - 1) / jmax + 1)
    lmax = jmax * imax

    cnt = np.zeros((int(lmax),), dtype=int)
    xcm = np.zeros((int(lmax),))
    ycm = np.zeros((int(lmax),))
    size = float(xbins)
    bnd = np.array([int(imax), int(jmax)])
    n = int(n)
    cellid = np.zeros((n,), dtype=int)

    # Constants for scaling the data
    xmin = xbnds[0]
    ymin = ybnds[0]
    xr = xbnds[1] - xmin
    yr = ybnds[1] - ymin
    c1 = size / xr
    c2 = size * shape / (yr * np.sqrt(3.))

    jinc = bnd[1]
    lat = jinc + 1
    iinc = 2 * jinc
    con1 = 0.25
    con2 = 1.0 / 3.0

    # Binning loop
    for i in range(n):
        sx = c1 * (x[i] - xmin)
        sy = c2 * (y[i] - ymin)
        j1 = int(np.floor(sx + 0.5))
        i1 = int(np.floor(sy + 0.5))
        dist1 = (sx - j1) ** 2 + 3. * (sy - i1) ** 2

        if dist1 < con1:
            l = i1 * iinc + j1 + 1
        elif dist1 > con2:
            l = int(sy) * iinc + int(sx) + lat
        else:
            j2 = int(np.floor(sx))
            i2 = int(np.floor(sy))
            if dist1 <= (sx - j2 - 0.5) ** 2 + 3. * (sy - i2 - 0.5) ** 2:
                l = i1 * iinc + j1 + 1
            else:
                l = i2 * iinc + j2 + lat

        cnt[l - 1] += 1
        cellid[i] = l

        xcm[l - 1] += (x[i] - xcm[l - 1]) / cnt[l - 1]
        ycm[l - 1] += (y[i] - ycm[l - 1]) / cnt[l - 1]

    nc = np.sum(cnt > 0)
    cell_out = np.where(cnt > 0)[0] + 1
    cnt_out = cnt[cnt > 0]
    xcm_out = xcm[cnt > 0]
    ycm_out = ycm[cnt > 0]
    n = nc
    bnd[0] = (cell_out[nc - 1] - 1) // bnd[1] + 1

    return {
        "cellid": cellid,
        "cell": cell_out,
        "count": cnt_out,
        "xcm": xcm_out,
        "ycm": ycm_out,
        "n": n,
        "bnd": bnd,
        "dimen": np.array([int(imax), int(jmax)])
    }


def calculate_hex_coords(shots, binwidths):
    """

    Args:
        shots:
        binwidths:

    Returns:

    """
    xbnds = hex_bounds(shots.loc_x, binwidths[0])  # MIN MAX по оси X
    xbins = np.diff(xbnds)[0] / binwidths[0]  # Кол-во бинов по оси X
    ybnds = hex_bounds(shots.loc_y, binwidths[1])  # MIN MAX по оси Y
    ybins = np.diff(ybnds)[0] / binwidths[1]  # Кол-во бинов по оси Y

    hb = hexbin(
        x=shots.loc[:, "loc_x"],
        y=shots.loc[:, "loc_y"],
        xbins=xbins,
        xbnds=xbnds,
        ybnds=ybnds,
        shape=ybins / xbins
    )

    shots["hexbin_id"] = hb["cellid"]

    hexbin_ids_to_zones = (
        shots
        .groupby(["hexbin_id", "shot_zone_range", "shot_zone_area"], as_index=False)
        .agg(attempts=pd.NamedAgg(column="hexbin_id", aggfunc="count"))
        .sort_values(by=["hexbin_id", "attempts"], ascending=[True, False])
        .assign(row_number=lambda df_: df_.groupby(["hexbin_id"]).cumcount() + 1)
        .pipe(lambda df_: df_.loc[df_.row_number == 1, ["hexbin_id", "shot_zone_range", "shot_zone_area"]])
        .reset_index(drop=True)
    )

    hexbin_stats = (
        shots
        .assign(points=lambda df_: df_.shot_made_flag * df_.shot_value)
        .groupby(["hexbin_id"], as_index=False)
        .agg(
            hex_attempts=pd.NamedAgg(column="hexbin_id", aggfunc="count"),
            hex_pct=pd.NamedAgg(column="shot_made_flag", aggfunc="mean"),
            hex_points_scored=pd.NamedAgg(column="points", aggfunc="sum"),
            hex_points_per_shot=pd.NamedAgg(column="points", aggfunc="mean")
        )
        .pipe(lambda df_: df_.merge(hexbin_ids_to_zones, how="inner", on="hexbin_id"))
    )

    sx = xbins / np.diff(xbnds)[0]
    sy = (xbins * ybins / xbins) / np.diff(ybnds)[0]
    dx = 1 / (2 * sx)
    dy = 1 / (2 * np.sqrt(3) * sy)

    origin_coords = {
        "x": np.repeat([dx, dx, 0, -dx, -dx, 0], 1),
        "y": np.repeat([dy, -dy, -2 * dy, -dy, dy, 2 * dy], 1)
    }

    ## hcell2xy(hb)
    c3 = np.diff(xbnds)[0] / xbins
    c4 = (np.diff(ybnds)[0] * np.sqrt(3)) / (2 * (ybins / xbins * xbins))
    jmax = hb["dimen"][1]
    cell = hb["cell"] - 1
    i = cell // jmax
    j = cell % jmax
    y = c4 * i + ybnds[0]
    x = [c3 * j_ + xbnds[0] if i_ % 2 == 0 else c3 * (j_ + 0.5) + xbnds[0] for j_, i_ in zip(j, i)]
    hex_centers = {
        "x": x,
        "y": y
    }

    hexbin_coords = pd.DataFrame(columns=["x", "y", "center_x", "center_y", "hexbin_id"])

    for i in range(len(hb["cell"])):
        cell_df = pd.DataFrame({
            "x": origin_coords["x"] + hex_centers["x"][i],
            "y": origin_coords["y"] + hex_centers["y"][i],
            "center_x": hex_centers["x"][i],
            "center_y": hex_centers["y"][i],
            "hexbin_id": hb["cell"][i]
        })
        hexbin_coords = pd.concat([hexbin_coords, cell_df], axis=0, ignore_index=True)

    return hexbin_coords.merge(hexbin_stats, how="inner", on="hexbin_id")


def calculate_hexbins_from_shots(shots, league_averages, binwidths=np.array([-1, 1]),
                                 min_radius_factor=0.6,
                                 fg_diff_limits=np.array([-0.12, 0.12]),
                                 fg_pct_limits=np.array([0.2, 0.7]),
                                 pps_limits=np.array([0.5, 1.5])):
    """

    Args:
        shots:
        league_averages:
        binwidths:
        min_radius_factor:
        fg_diff_limits:
        fg_pct_limits:
        pps_limits:

    Returns:

    """
    zone_stats = (
        shots
        .assign(points=lambda df_: df_.shot_made_flag * df_.shot_value)
        .groupby(["shot_zone_range", "shot_zone_area"], as_index=False)
        .agg(
            zone_attempts=pd.NamedAgg(column="loc_x", aggfunc="count"),
            zone_pct=pd.NamedAgg(column="shot_made_flag", aggfunc="mean"),
            zone_points_scored=pd.NamedAgg(column="points", aggfunc="sum"),
            zone_points_per_shot=pd.NamedAgg(column="points", aggfunc="mean")
        )
    )

    league_zone_stats = (
        league_averages
        .groupby(["shot_zone_range", "shot_zone_area"], as_index=False)
        .agg(
            sum_fgm=("fgm", sum),
            sum_fga=("fga", sum)
        )
        .assign(league_pct=lambda df_: df_.sum_fgm / df_.sum_fga)
        .drop(columns=["sum_fgm", "sum_fga"])
    )

    hex_data = calculate_hex_coords(shots, binwidths=binwidths)
    max_hex_attempts = np.max(hex_data.loc[:, "hex_attempts"])

    hex_data = (
        hex_data
        .pipe(lambda df_: df_.merge(zone_stats, how="inner", on=["shot_zone_area", "shot_zone_range"]))
        .pipe(lambda df_: df_.merge(league_zone_stats, how="inner", on=["shot_zone_area", "shot_zone_range"]))
        .assign(
            radius_factor=lambda df_: min_radius_factor + (1 - min_radius_factor) * np.log(
                df_.hex_attempts + 1) / np.log(max_hex_attempts + 1),
            adj_x=lambda df_: df_.center_x + df_.radius_factor * (df_.x - df_.center_x),
            adj_y=lambda df_: df_.center_y + df_.radius_factor * (df_.y - df_.center_y),
            bounded_fg_diff=lambda df_: np.clip(df_.zone_pct - df_.league_pct, fg_diff_limits[0], fg_diff_limits[1]),
            bounded_fg_pct=lambda df_: np.clip(df_.zone_pct, fg_pct_limits[0], fg_pct_limits[1]),
            bounded_points_per_shot=lambda df_: np.clip(df_.zone_points_per_shot, pps_limits[0], pps_limits[1])
        )
    )

    return {
        "hex_data": hex_data,
        "fg_diff_limits": fg_diff_limits,
        "fg_pct_limits": fg_pct_limits,
        "pps_limits": pps_limits
    }
