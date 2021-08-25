/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;
using System.Windows.Media.Media3D;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// This class is for the parameter g file in constant folder.
    /// </summary>
    public class G : FOAMDict
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this file.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary</param>
        /// <param name="settings">Settings-object</param>
        public G(Version version, string path, Dictionary<string, object> attributes, SaveFormat format, Settings settings)
            : base("g", "uniformDimensionedVectorField", version, path, attributes, format)
        {
            InitAttributes();
        }

        /// <summary>
        /// Initialize Attributes.
        /// </summary>
        public override void InitAttributes()
        {
            int[] dimension = new int[] { 0, 1, -2, 0, 0, 0, 0 };
            FoamFile.Attributes.Add("dimensions", dimension);
            FoamFile.Attributes.Add("value", new Vector3D(0, 0, (double)m_DictFile["g"]/*BIM.OpenFOAMExport.Exporter.Instance.settings.GValue*/));
        }
    }
}
