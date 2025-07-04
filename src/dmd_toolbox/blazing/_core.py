import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dmd_toolbox.utils import checkdep_usetex

# workaround to allow to run in interactive window
ROOT = Path(os.getcwd())
if all(item in ROOT.parts for item in ["src", "dmd_toolbox"]):
    ROOT = ROOT.parent.parent

def unitVecSphericalToCartesianEquator(lat, lon):
    """ "
    Spherical coordinates to Cartesian coordinates conversion.
    Latitude and longitude are defined as in the DMD schematics in the Notion page.
    Parameters:
    ---------
    lat : float
        Latitude in radians, from -π/2 to π/2.
    lon : float
        Longitude in radians, from -π to π.

    Returns:
    --------
    np.array
        A 3D vector in Cartesian coordinates (x, y, z).
    The z-axis is the normal to the DMD,
    the x-axis is the direction of the tilt of the mirrors in the ON state, and the y-axis is perpendicular to both.
    The coordinates are ordered as (x, y, z).
    The x-axis is the direction of the tilt of the mirrors in the ON state, and the y-axis is perpendicular to both.
    The coordinates are ordered as (x, y, z).
    """
    return np.array([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)])


def unitVecSphereToCartesianNormal(
    theta, phi
):  # Z in on top (as Y, in the first function)
    """Converts spherical coordinates (theta, phi) to Cartesian coordinates.
    Parameters:
    ----------
    theta : float
        Polar angle in radians, from 0 to π.
    phi : float
        Azimuthal angle in radians, from 0 to 2π.

    Returns:
    -------
    np.array
        A 3D vector in Cartesian coordinates (x, y, z).
    """
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def getReflectedVector(normal, incident):
    """ "Calculates the reflected vector about the normal vector to the plane interface, given the incident vector.
    Parameters:
    ----------
    normal : np.array
        A 3D vector representing the normal to the surface.
    incident : np.array
        A 3D vector representing the incident wave vector.

    Returns:
    -------
    np.array
        A 3D vector representing the reflected wave vector.
    """
    return incident - 2 * np.dot(incident, normal) * normal


def _osc(i):
    return 0.1 if i % 2 == 0 else 0


THRESH = 0.2


class DMDSetup:
    """ "
    Class that models a DMD illuminated by a beam of a given wavlength.
    Default values for the mirror pitch,
    the tilt angle and the tilt direction are set to match those of the DLP6500FLQ DMD.
    The DMD modelled here with the clipped corner of its evulation module PCB in the top right corner.
    """

    def __init__(
        self,
        theta_tilt: float = np.deg2rad(12),
        tilt_direction: float = np.deg2rad(45 + 90),
        mirror_pitch: float = 7.56e-6,
        wavelength: float = 638e-9,
        path_file: str | Path = "Plots",
        custom_lat: float | None = None,
    ):
        """ "Initializes the DMD setup with the given parameters.
        Parameters
        ----------
        custom_lat: float | None (radians)
            Custom latitude in degrees for the planar angle (default is None, which will use the default tilt angle).
        path_file: str, path
            Path to save the results (default is in the directory "Plots" that will be created in the root folder).
            The path should be taken from project root.

        The following parameters' defaults are set to match those of the DLP6500FLQ DMD:
        theta_tilt: float (radians)
            Tilt angle of the mirrors in the ON state, in radians (default is 12 degrees).
        tilt_direction: float (radians)
            Tilt direction of the mirrors in the ON state, in radians
            (default is 135 degrees, i.e. towards the clipped corner of the DMD PCB).
        mirror_pitch: float (meters)
            Pitch of the mirrors in meters (default is 7.56 µm).
        wavelength: float (meters)
            Wavelength of the incident beam, in m (default is 638 nm).
        Raises
        ------
        - ValueError: If the custom latitude is not between -90° and 90° (in radians: -π/2 to π/2).
        """

        self.theta_tilt = theta_tilt
        self.tilt_direction = tilt_direction
        self.mirror_pitch = mirror_pitch
        self.wavelength = wavelength
        self.normal_mirror_ON = unitVecSphereToCartesianNormal(
            self.theta_tilt, self.tilt_direction
        )
        self.path_file = ROOT.joinpath(path_file)

        self.latlong_array = np.linspace(-np.pi / 2, np.pi / 2, 500)
        if custom_lat is not None:
            self.lat_planar = custom_lat
            if not (-np.pi / 2 <= self.lat_planar <= np.pi / 2):
                raise ValueError(
                    "Custom latitude must be between -π/2 to π/2 in radians."
                )
            self.lat_planar = np.deg2rad(self.lat_planar)
        else:
            self.lat_planar = np.arctan(
                np.tan(self.theta_tilt) / np.sqrt(2)
            )  # in rad - angle such that k_i and k_r are coplanar with the normal to the mirror in the ON state
        self.idx_planar = np.argmin(np.abs(self.latlong_array - self.lat_planar))

    def get_phase_shift_XY(self, k_i):
        """ "
        Returns the phase shift to a diffracted order (mx, my) in the x and y directions
        for a given incident wave vector k_i and the normal vector of the mirror.
        Parameters
        ----------
        k_i: float
            Incident wave vector, a 3D vector in Cartesian coordinates.
        Returns
        -------
        delta_k_x: float
            Phase shift in the x direction, normalised by the DMD spatial frequency (units of the DMD pitch).
        delta_k_y: float
            Phase shift in the y direction, normalised by the DMD spatial frequency (units of the DMD pitch).
        mx: float
            Closest inferior order in the x direction.
        my: float
            Closest inferior order in the y direction.
        """
        k_r_mirror = getReflectedVector(self.normal_mirror_ON, k_i)

        delta_k = (k_i - k_r_mirror) * self.mirror_pitch / self.wavelength

        # Along the x axis
        delta_k_x = delta_k[0]
        mx = np.round(delta_k_x)
        delta_k_x = np.abs(delta_k_x - mx)

        # Along the y axis
        delta_k_y = delta_k[1]
        my = np.round(delta_k_y)
        delta_k_y = np.abs(delta_k_y - my)

        return delta_k_x, delta_k_y, mx, my

    def get_all_phase_shifts(self):
        """Computes all the quadratic phase shifts for all k_i vectors."""
        # Compute phase shifts for any k_i (latitude_i, longitude_i)
        self.phase_shifts = np.zeros((len(self.latlong_array), len(self.latlong_array)))
        self.orderX = np.zeros((len(self.latlong_array), len(self.latlong_array)))
        self.orderY = np.zeros((len(self.latlong_array), len(self.latlong_array)))

        for i in range(len(self.latlong_array)):  # iterate on the latitude_i
            for j in range(len(self.latlong_array)):  # iterate on the longitude_i
                k_i = unitVecSphericalToCartesianEquator(
                    self.latlong_array[i], self.latlong_array[j]
                )
                delta_k_x, delta_k_y, mx, my = self.get_phase_shift_XY(k_i)
                self.phase_shifts[i, j] = np.sqrt(delta_k_x**2 + delta_k_y**2)
                self.orderX[i, j] = mx
                self.orderY[i, j] = my

        phaseShiftXYSingleLat = self.phase_shifts[
            self.idx_planar, :
        ]  # phase shift for a given latitude (the planar one)

        # Solve minima for the phase shift, and store in blazed_longs_indices
        self.blazed_longs_indices = []
        for i in range(1, len(phaseShiftXYSingleLat) - 1):
            if (
                phaseShiftXYSingleLat[i - 1] > phaseShiftXYSingleLat[i]
                and phaseShiftXYSingleLat[i] < phaseShiftXYSingleLat[i + 1]
                and phaseShiftXYSingleLat[i] < THRESH
            ):  # i.e. local minimum
                self.blazed_longs_indices.append(i)

        self.blazed_orders = [
            (
                float(self.orderX[self.idx_planar, self.blazed_longs_indices[i]]),
                float(self.orderY[self.idx_planar, self.blazed_longs_indices[i]]),
            )
            for i in range(len(self.blazed_longs_indices))
        ]
        print(
            f"Found {len(self.blazed_orders)} blazed orders for lat = {np.rad2deg(self.lat_planar):.2f}°."
        )

    def plot_phase_shift(self):
        """Plots the phase shifts for all k_i vectors and for a given latitude.
        The first plot is the phase shift for any incident longitude or latitude,
        and the second plot is the phase shift for any longitude at latitude `lat_planar`.
        The blazed orders are marked with vertical dashed lines on both plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(
            r"Phase Shifts for $\theta_{tilt}=$"
            + f" ${np.rad2deg(self.theta_tilt):.1f}^{{\\circ}}$"
            + r" , $\lambda =$"
            + f" ${self.wavelength * 10**9:.0f}$"
            + " $\\mathrm{nm}$"
            + r" and $d_{mirror} = $"
            + f"${self.mirror_pitch * 1e6:.2f}$"
            + " $\\mathrm{\\mu m}$",
            fontsize="20"
        )
        # Plot the phase shifts
        ax[0].imshow(self.phase_shifts, extent=[-90, 90, 90, -90], cmap="Greys")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(ax[0].images[0], cax=cax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Phase Shift (a.u.)", fontsize=12)
        ax[0].set_xlabel("Longitude $\\mathrm{lon}_i$ (deg)", fontsize=14)
        ax[0].set_ylabel("Latitude $\\mathrm{lat}$ (deg)", fontsize=14)
        ax[0].set_title("Phase Shift for any incident beam", fontsize=14)
        ax[0].axhline(
            y=np.rad2deg(self.lat_planar),
            color="darkolivegreen",
            linestyle="--",
            label="Tilt Direction",
        )
        ax[0].text(
            self.theta_tilt - 20,
            np.rad2deg(self.lat_planar) - 2,
            r"$\mathbf{\mathrm{lat}_{i}=}$"
            + f" ${np.rad2deg(self.lat_planar):.2f}^{{\\circ}}$",
            color="white",
            fontsize=17
        )

        for i in range(len(self.blazed_longs_indices)):
            ax[0].axvline(
                x=np.rad2deg(self.latlong_array[self.blazed_longs_indices[i]]),
                color="maroon",
                linestyle="--",
                label=f"{self.blazed_orders[i]}",
            )
            # print(np.rad2deg(latlong_array[blazed_longs_indices[i]]))

        ## Plot for one given latitude
        ax[1].plot(
            np.rad2deg(self.latlong_array),
            self.phase_shifts[self.idx_planar, :],
            color="slategray",
        )
        ax[1].set_xlabel("Longitude $\\mathrm{lon}_i$ (deg)", fontsize=14)
        ax[1].set_ylabel("Phase Shift (a.u.)", fontsize=14)
        ax[1].set_title(
            "Phase Shift for "
            + r"$\mathrm{lat}_{i}$ = "
            + f"${np.rad2deg(self.lat_planar):.2f}^{{\\circ}}$",
            fontsize=14,
        )
        ax[1].grid()

        # Add vertical lines for blazed orders
        for i in range(len(self.blazed_longs_indices)):
            ax[1].axvline(
                x=np.rad2deg(self.latlong_array[int(self.blazed_longs_indices[i])]),
                color="maroon",
                linestyle="--",
                label=f"{self.blazed_orders[i]}",
            )
            ax[1].text(
                np.rad2deg(self.latlong_array[self.blazed_longs_indices[i]]),
                self.phase_shifts[self.idx_planar, :][self.blazed_longs_indices[i]]
                + _osc(i),
                f"{self.blazed_orders[i]} \n "
                + r"$\mathrm{lon}_{i} =$"
                + f"${np.rad2deg(self.latlong_array[self.blazed_longs_indices[i]]):.2f}^{{\\circ}}$",
                color="maroon",
                fontsize=12,
            )

        # Save to pathfile
        # Check if the path exists, if not create it
        if not os.path.exists(self.path_file):
            os.makedirs(self.path_file)


        plt.savefig(self.path_file.joinpath("Phase_Shifts.png"), dpi=300, bbox_inches="tight")


# if __name__ == "__main__":
