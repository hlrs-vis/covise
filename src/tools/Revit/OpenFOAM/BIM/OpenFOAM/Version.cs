/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
namespace BIM.OpenFOAMExport.OpenFOAM
{
    /// <summary>
    /// Represents the OpenFoam-Version.
    /// </summary>
    public class Version
    {
        /// <summary>
        /// Full version of openFoam.
        /// </summary>
        public string OFFullVer { set; get; } = "v1612+";

        /// <summary>
        /// General version of openFoam.
        /// </summary>
        public string OFVer { set; get; } = "4.0";

        /// <summary>
        /// Addin-Version.
        /// </summary>
        public string AddinVer { set; get; } = "0.0.1";

        /// <summary>
        /// Header object.
        /// </summary>
        public Header Header { set; get; }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">blueCFD, linux, docker(windows)</param>
        public Version(OpenFOAMEnvironment env = OpenFOAMEnvironment.blueCFD)
        {
            SetVersionsByEnv(env);
            Header = new Header(this);
        }

        /// <summary>
        /// TO-DO: Set version if there are new Versions of blueCFD.
        /// </summary>
        /// <param name="env"></param>
        private void SetVersionsByEnv(OpenFOAMEnvironment env)
        {
            if(env == OpenFOAMEnvironment.blueCFD)
            {

            }
            else
            {

            }
        }

    }

    /// <summary>
    /// Header for OpenFoam-Files.
    /// </summary>
    public class Header
    {
        /// <summary>
        /// Represents the header of each dictionary.
        /// </summary>
        public string HeaderStr { get; }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="version">Version-Object.</param>
        public Header(Version version)
        {
            HeaderStr = string.Format(
                "/*--------------------------------*- C++ -*----------------------------------*\\\n" +
                "| =========                 |                                                 |\n" +
                "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n" +
                "|  \\\\    /   O peration     | Version:  {0}                                |\n" +
                "|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n" +
                "|    \\\\/     M anipulation  |                                                 |\n" +
                "\\*---------------------------------------------------------------------------*/\n" +
                "/* Revit-OpenFOAM-Addin {1}                 *\\\n" +
                "\\*---------------------------------------------------------------------------*/\n", 
                version.OFFullVer, version.AddinVer); 
        }
    }
}
