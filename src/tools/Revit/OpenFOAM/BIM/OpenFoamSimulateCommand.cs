/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System.IO;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace OpenFOAMInterface.BIM
{
    /// <summary>
    /// Class OpenFOAMSimulateCommand is the entry of the AddIn program.
    /// </summary>
    [Regeneration(RegenerationOption.Manual)]
    [Transaction(TransactionMode.Manual)]

    public class OpenFOAMSimulateCommand : IExternalCommand
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
            //for repeating click-events

            m_Revit = commandData.Application;

            Exporter.Instance.settings.SetDocument(m_Revit);
            string fileName = "wallSTL.stl";
            // save Revit document's triangular data in a temporary file, generate openFOAM-casefolder and start simulation

            Directory.CreateDirectory(Exporter.Instance.settings.LocalCaseFolder);
            Directory.CreateDirectory(Exporter.Instance.settings.LocalCaseFolder + "\\constant");
            Directory.CreateDirectory(Exporter.Instance.settings.LocalCaseFolder + "\\constant\\triSurface");

            DataGenerator Generator = new(m_Revit.Application, m_Revit.ActiveUIDocument.Document);
            DataGenerator.GeneratorStatus succeed = Generator.SaveSTLFile(fileName);
            if (succeed == DataGenerator.GeneratorStatus.SUCCESS)
            {
                Exporter.Instance.settings.InitConfigs();
                succeed = Generator.CreateOpenFOAMCase(Exporter.Instance.settings.LocalCaseFolder);
            }

            if (succeed == DataGenerator.GeneratorStatus.SUCCESS)
                return Result.Succeeded;
            return Result.Failed;
        }

    }
}
