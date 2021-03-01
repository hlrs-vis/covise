/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.Threading.Tasks;
using System.Windows.Forms;
using utils;

using Autodesk.Revit;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace BIM.OpenFOAMExport
{
    /// <summary>
    /// Class OpenFOAMExportCommand is the entry of the AddIn program.
    /// </summary>
    [Regeneration(RegenerationOption.Manual)]
    [Transaction(TransactionMode.Manual)]
    
    public class OpenFOAMExportCommand : IExternalCommand
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
            BIM.OpenFOAMExport.Exporter.Instance.exportForm = new OpenFOAMExportForm(m_Revit);
            BIM.OpenFOAMExport.Exporter.Instance.settings.setDocument(m_Revit);

            //for repeating click-events
            System.Collections.IEnumerator iterator = System.Windows.Forms.Application.OpenForms.GetEnumerator();
            while(iterator.MoveNext())
            {
                System.Windows.Forms.Form form = iterator.Current as System.Windows.Forms.Form;
                if(form is OpenFOAMExportForm)
                {
                    return Result.Succeeded;
                }
            }
            Result result = StartOpenFOAMExportForm();
            return result;

        }

        /// <summary>
        /// Generates OpenFOAMExportForm and shows it.
        /// </summary>
        private Result StartOpenFOAMExportForm()
        {
            if (m_Revit == null)
                return Result.Failed;

            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            System.Windows.Forms.Application.EnableVisualStyles();
            //exportForm.TopMost = true;

            //Start modal form with with responsive messageloop.
            System.Windows.Forms.Application.Run(BIM.OpenFOAMExport.Exporter.Instance.exportForm);

            if (BIM.OpenFOAMExport.Exporter.Instance.exportForm.DialogResult == DialogResult.Cancel)
            {
                return Result.Cancelled;
            }

            return Result.Succeeded;
        }
    }
}
