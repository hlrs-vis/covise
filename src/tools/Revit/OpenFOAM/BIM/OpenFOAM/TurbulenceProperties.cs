/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// This class is used for initialize the turbulence properties for the OpenFOAM simulation.
    /// </summary>
    public class TurbulenceProperties : FOAMDict
    {
        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this File.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary.</param>
        /// <param name="settings">Data-objects</param>
        public TurbulenceProperties(Version version, string path, Dictionary<string, object> attributes, SaveFormat format)
            : base("turbulenceProperties", "dictionary", version, path, attributes, format)
        {
            InitAttributes();
        }

        /// <summary>
        /// Initialize all attributes.
        /// </summary>
        public override void InitAttributes()
        {
            foreach(var obj in m_DictFile)
            {
                FoamFile.Attributes.Add(obj.Key, obj.Value);
            }
        }
    }
}
