/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;

namespace BIM.OpenFOAMExport.OpenFOAM
{
    /// <summary>
    /// The DecomposParDict-Class contains all attributes for decomposing case for parallel computing in Openfoam.
    /// </summary>
    public class DecomposeParDict : FOAMDict
    {
        /// <summary>
        /// Number of cpu cores
        /// </summary>        
        private int m_NumberOfSubdomains;
        
        /// <summary>
        /// Getter for numberOfSubdomains
        /// </summary>
        public int NumberOfSubdomains { get => m_NumberOfSubdomains;}

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this File.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary.</param>
        /// <param name="settings">Settings-objects</param>
        public DecomposeParDict(Version version, string path, Dictionary<string, object> attributes, SaveFormat format)
            : base("decomposeParDict", "dictionary", version, path, attributes, format)
        {
            //m_SimpleCoeffs = new Dictionary<string, object>();
            //m_HierarchicalCoeffs = new Dictionary<string, object>();
            //m_ManualCoeffs = new Dictionary<string, object>();
            InitAttributes();
        }

        /// <summary>
        /// Initialize attributes.
        /// </summary>
        public override void InitAttributes()
        {
            m_NumberOfSubdomains = BIM.OpenFOAMExport.Exporter.Instance.settings.NumberOfSubdomains;
            FoamFile.Attributes.Add("numberOfSubdomains", m_NumberOfSubdomains);
            base.InitAttributes();
        }

    }
}
