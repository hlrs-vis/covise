/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// This class is in use to calculate simulation parameter.
    /// </summary>
    public class OpenFOAMCalculator
    {
        /// <summary>
        /// Constant for dissipation rate epsilon.
        /// </summary>
        private const double m_Cnue = 0.09;

        /// <summary>
        /// Constant for kinematic viscosity for air at atmospheric pressure of 1 bar and 20 degree.
        /// Source:http://www.uni-magdeburg.de/isut/LSS/Lehre/Arbeitsheft/IV.pdf
        /// </summary>
        private static double m_ViAir20 = 153.5 * Math.Pow(10, -7);

        /// <summary>
        /// Constant for kinematic viscosity for air at atmospheric pressure of 1 bar and 30 degree.
        /// Source: http://www.uni-magdeburg.de/isut/LSS/Lehre/Arbeitsheft/IV.pdf
        /// </summary>
        private static double m_ViAir30 = 163 * Math.Pow(10, -7);

        /// <summary>
        /// Construcor.
        /// </summary>
        public OpenFOAMCalculator()
        {
        }

        /// <summary>
        /// Calculate k in OpenFOAM-convention for k-epsilon turbulencemodel.
        /// Source: https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-epsilon.html
        /// </summary>
        /// <param name="meanFlowVelocity">Mean flow velocity.</param>
        /// <param name="turbulenceIntensity">Turbulenc intensity.</param>
        /// <returns>k for k-epsilon Turbulencemodel.</returns>
        public double CalculateK(double meanFlowVelocity, double turbulenceIntensity) => Math.Pow(meanFlowVelocity * turbulenceIntensity, 2) * 3 / 2;

        /// <summary>
        /// Calculate epsilon in OpenFOAM-convetion for k-epsilon turbulencemodel.
        /// Source: https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-epsilon.html
        /// </summary>
        /// <param name="turbulenceLengthScale">estimated size for turbulent eddies.</param>
        /// <param name="k">Turbulence energie.</param>
        /// <returns>Epsilon for k-epsilon turbulencemodel.</returns>
        public double CalculateEpsilon(double turbulenceLengthScale, double k) => Math.Pow(m_Cnue, 0.75) * Math.Pow(k, 1.5) / turbulenceLengthScale;

        /// <summary>
        /// Calculates Reynoldsnumber.
        /// </summary>
        /// <param name="meanFlowVelocity">Mean flow velocity.</param>
        /// <param name="kinematicViscosity">Kinematic viscosity of the fluid.</param>
        /// <param name="characteristicLength">Characteristic length (hydraulic diameter in pipes)</param>
        /// <returns>Reynoldsnumber as double.</returns>
        public double CalculateReynoldsnumber(double meanFlowVelocity, double kinematicViscosity, double characteristicLength) => meanFlowVelocity * characteristicLength / kinematicViscosity;

        /// <summary>
        /// Calculate the hydraulicDiameter for pipes that aren't round.
        /// </summary>
        /// <param name="area">Flow area in pipe.</param>
        /// <param name="boundaryLength">Boundary length of flow area.</param>
        /// <returns>Hydraulic diameter for unround pipes.</returns>
        public double CalculateHydraulicDiameter(double area, double boundaryLength) => 4 * area / boundaryLength;

        /// <summary>
        /// Estimate the turbulence intensity based on Reynoldsnumber.
        /// for fully developed piped flows.
        /// Source: https://www.cfd-online.com/Wiki/Turbulence_intensity
        /// </summary>
        /// <param name="reynoldsNumber">Reynoldsnumber as double.</param>
        /// <returns>Turbulenceintensity as double.</returns>
        public double EstimateTurbulenceIntensityPipe(double reynoldsNumber) => 0.16 * Math.Pow(reynoldsNumber, -1.0 / 8.0);

        /// <summary>
        /// Returns the rho-normalized pressure.
        /// </summary>
        /// <param name="externalPressure">External pressure.</param>
        /// <param name="rho">Density of fluid.</param>
        /// <returns>rho-normalized pressure as double.</returns>
        public double CalculateRhoNormalizedPressure(double externalPressure, double rho) => externalPressure / rho;

        /// <summary>
        /// Estimate turbulence length scale for pipe flows.
        /// For fully developed piped flows.
        /// Source: https://www.cfd-online.com/Wiki/Turbulence_length_scale
        /// </summary>
        /// <param name="hydraulicDiameter">Hydraulic Diameter of the pipe.</param>
        /// <returns>Turbulence length scale as double.</returns>
        public double EstimateTurbulencLengthScalePipe(double hydraulicDiameter) => 0.038 * hydraulicDiameter;

        /// <summary>
        /// Interpolates kinematic vicosity for the given temprature between 20 and 30 degree.
        /// </summary>
        /// <param name="temp">Reference tempreture for kinematic viscosity.</param>
        /// <returns>Kinematic viscosity at temprature.</returns>
        public double InterpolateKinematicViscosity(double temp) => m_ViAir20 + (m_ViAir30 - m_ViAir20) * (temp - 20) / 10;
    }
}