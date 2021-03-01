/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace BIM.OpenFOAMExport
{
    /// <summary>
    /// Class OpenFOAMExportCommand is the entry of the AddIn program. Here its the direct export version.
    /// </summary>
    [Regeneration(RegenerationOption.Manual)]
    [Transaction(TransactionMode.Manual)]

    public class OpenFOAMExportButtonCommand : IExternalCommand
    {
        /// <summary>
        /// The application object for the active instance of Autodesk Revit.
        /// </summary>
        private UIApplication m_Revit;

        /// <summary>
        /// Implement the member of IExternalCommand Execute.
        /// </summary>
        /// <param name="commandData">
        /// The application object for the active instance of Autodesk Revit.
        /// </param>
        /// <param name="message">
        /// A message that can be set by the external command and displayed in case of error.
        /// </param>
        /// <param name="elements">
        /// A set of elements that can be displayed if an error occurs.
        /// </param>
        /// <returns>
        /// A value that signifies if yout command was successful, failed or the user wishes to cancel.
        /// </returns>
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            m_Revit = commandData.Application;
            Result result = StartOpenFOAMExportFormButtonDirect();
            return result;
        }
        
        /// <summary>
        /// Generates OpenFOAMExportForm without showing it and starts simulation directly.
        /// </summary>
        private Result StartOpenFOAMExportFormButtonDirect()
        {
            if (m_Revit == null)
                return Result.Failed;

            using (OpenFOAMExportForm exportForm = new OpenFOAMExportForm(m_Revit, true))
            {
                if (exportForm.DialogResult == DialogResult.Cancel)
                {
                    return Result.Cancelled;
                }
            }

            return Result.Succeeded;
        }
    }
}
