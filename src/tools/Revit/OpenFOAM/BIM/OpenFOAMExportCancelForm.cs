/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.Windows.Forms;

namespace OpenFOAMInterface.BIM
{
    public partial class OpenFOAMExportCancelForm : Form
    {
        private bool m_CancelProcess = false;

        public bool CancelProcess
        {
            get { return m_CancelProcess; }
            set { m_CancelProcess = value; }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        public OpenFOAMExportCancelForm()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Cancel button click event.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The event args.</param>
        private void BtnCancel_Click(object sender, EventArgs e)
        {
            m_CancelProcess = !m_CancelProcess;

            if (!m_CancelProcess)
                Close();
        }
    }
}
