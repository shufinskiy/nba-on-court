from typing import Callable, Sequence, Union, Dict

import numpy as np
import pandas as pd


BINWIDTHS = np.array([1.5, 1.5])


def _round_any(x: float, accuracy: float, f: Callable=round) -> float:
    """
    Rounds a given floating-point number to a specified accuracy using a custom rounding function.

    Args:
        x (float): The numeric value to be rounded.
        accuracy (float): The target accuracy level to which `x` should be rounded.
        f (Callable): A callable object that performs the actual rounding operation. Defaults to Python's built-in `round` function.

    Returns:
        float: The rounded value, adjusted to the specified accuracy.
    """
    return f(x / accuracy) * accuracy


def _hex_bounds(x: Sequence, binwidth: float) -> np.ndarray:
    """
    Calculates the boundary values for constructing hexagonal bins.

    Args:
        x (Sequence): The data sequence upon which hexagon boundaries are based.
        binwidth (float): The width of each bin used in the hexagonal grid.

    Returns:
        np.ndarray: An array containing the minimum and maximum boundary values
                    for creating hexagonal bins, with adjustments made to ensure proper alignment.
    """
    return np.array([_round_any(np.min(x), binwidth, np.floor) - 1e-6,
                     _round_any(np.max(x), binwidth, np.ceil) + 1e-6])


def _hexbin(x: Union[pd.Series, Sequence],
           y: Union[pd.Series, Sequence],
           xbins: float,
           xbnds: Sequence,
           ybnds: Sequence,
           shape: float) -> dict:
    """
    Generates a hexagonal binning of input data points.

    Args:
        x (Union[pd.Series, Sequence]): The x-coordinates of the data points.
        y (Union[pd.Series, Sequence]): The y-coordinates of the data points.
        xbins (float): The number of bins along the x-axis.
        xbnds (Sequence): A tuple representing the lower and upper bounds of the x-axis.
        ybnds (Sequence): A tuple representing the lower and upper bounds of the y-axis.
        shape (float): The shape parameter of the hexagon (typically between 0 and 1).

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - 'cellid': An array of integers representing the bin index for each data point.
            - 'cell': An array of integers representing the non-empty bin indices.
            - 'count': An array of integers representing the count of data points in each bin.
            - 'xcm': An array of floats representing the x-coordinate of the center of mass for each bin.
            - 'ycm': An array of floats representing the y-coordinate of the center of mass for each bin.
            - 'n': An integer representing the total number of non-empty bins.
            - 'bnd': An array of two integers representing the dimensions of the hexagonal grid.
            - 'dimen': An array of two integers representing the effective dimensions of the grid.
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


def calculate_hex_coords(shots: pd.DataFrame, binwidths: Sequence) -> pd.DataFrame:
    """
    Calculates hexagonal coordinates and statistics for shot data.

    Args:
        shots (pd.DataFrame): DataFrame containing shot data.
        binwidths (Sequence): A sequence of two floats representing the bin widths for the x and y axes.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated hexagonal coordinates and statistics including:
            - 'x': The x-coordinate of the hexagon.
            - 'y': The y-coordinate of the hexagon.
            - 'center_x': The x-coordinate of the center of the hexagon.
            - 'center_y': The y-coordinate of the center of the hexagon.
            - 'hexbin_id': The ID of the hexagonal bin.
            - 'hex_attempts': Total attempts within the hexagonal bin.
            - 'hex_pct': Percentage of successful shots within the hexagonal bin.
            - 'hex_points_scored': Total points scored within the hexagonal bin.
            - 'hex_points_per_shot': Average points per shot within the hexagonal bin.
            - 'shot_zone_range': The shot zone range associated with the hexagonal bin.
            - 'shot_zone_area': The shot zone area associated with the hexagonal bin.
    """
    xbnds = _hex_bounds(shots.loc_x, binwidths[0])
    xbins = np.diff(xbnds)[0] / binwidths[0]
    ybnds = _hex_bounds(shots.loc_y, binwidths[1])
    ybins = np.diff(ybnds)[0] / binwidths[1]

    hb = _hexbin(
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


def calculate_hexbins_from_shots(shots: pd.DataFrame,
                                 league_averages: pd.DataFrame,
                                 binwidths: Sequence=np.array([1, 1]),
                                 min_radius_factor: int=0.6,
                                 fg_diff_limits: Sequence=np.array([-0.12, 0.12]),
                                 fg_pct_limits: Sequence=np.array([0.2, 0.7]),
                                 pps_limits: Sequence=np.array([0.5, 1.5])) -> Dict[str, Union[pd.DataFrame, Sequence]]:
    """
    Calculates hexagonal bin statistics and adjusts them based on shot zone data and league averages.

    Args:
        shots (pd.DataFrame): DataFrame containing shot data with columns.
        league_averages (pd.DataFrame): DataFrame containing league average shooting statistics.
        binwidths (Sequence, optional): A sequence of two floats representing the bin widths for the x and y axes.
                                        Defaults to `np.array([-1, 1])`.
        min_radius_factor (int, optional): Minimum radius factor for adjusting hexagon sizes. Defaults to `0.6`.
        fg_diff_limits (Sequence, optional): Limits for clipping the difference between zone and league FG percentage.
                                            Defaults to `np.array([-0.12, 0.12])`.
        fg_pct_limits (Sequence, optional): Limits for clipping the FG percentage within the zone.
                                            Defaults to `np.array([0.2, 0.7])`.
        pps_limits (Sequence, optional): Limits for clipping the points per shot within the zone.
                                        Defaults to `np.array([0.5, 1.5])`.

    Returns:
        Dict[str, Union[pd.DataFrame, Sequence]]: A dictionary containing the following keys:
            - 'hex_data': A DataFrame containing the calculated hexagonal bin statistics including:
                - 'x': The x-coordinate of the hexagon.
                - 'y': The y-coordinate of the hexagon.
                - 'center_x': The x-coordinate of the center of the hexagon.
                - 'center_y': The y-coordinate of the center of the hexagon.
                - 'hexbin_id': The ID of the hexagonal bin.
                - 'hex_attempts': Total attempts within the hexagonal bin.
                - 'hex_pct': Percentage of successful shots within the hexagonal bin.
                - 'hex_points_scored': Total points scored within the hexagonal bin.
                - 'hex_points_per_shot': Average points per shot within the hexagonal bin.
                - 'shot_zone_range': The shot zone range associated with the hexagonal bin.
                - 'shot_zone_area': The shot zone area associated with the hexagonal bin.
                - 'radius_factor': Adjusted radius factor for each hexagon.
                - 'adj_x': Adjusted x-coordinate after applying the radius factor.
                - 'adj_y': Adjusted y-coordinate after applying the radius factor.
                - 'bounded_fg_diff': Clipped difference between zone and league FG percentage.
                - 'bounded_fg_pct': Clipped FG percentage within the zone.
                - 'bounded_points_per_shot': Clipped points per shot within the zone.
            - 'fg_diff_limits': The limits for clipping the difference between zone and league FG percentage.
            - 'fg_pct_limits': The limits for clipping the FG percentage within the zone.
            - 'pps_limits': The limits for clipping the points per shot within the zone.
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
