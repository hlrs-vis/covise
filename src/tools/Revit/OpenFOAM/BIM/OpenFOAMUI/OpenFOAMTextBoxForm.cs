/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace OpenFOAMInterface.BIM.OpenFOAMUI
{
    /// <summary>
    /// Class that uses Form as base class combined with a textBox.
    /// </summary>
    public partial class OpenFOAMTextBoxForm : Form
    {
        /// <summary>
        /// State of the form.
        /// </summary>
        private bool m_CancelProcess = false;

        /// <summary>
        /// TextBox.
        /// </summary>
        private TextBox m_TxtBox = new TextBox();

        /// <summary>
        /// Regular expression for text in textBox.
        /// </summary>
        private readonly Regex m_RegTxt;

        #region Constructor.

        /// <summary>
        /// Default constructor.
        /// </summary>
        public OpenFOAMTextBoxForm()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Initialize Regular Expression with this constructor.
        /// </summary>
        /// <param name="reg">Regex object.</param>
        /// <param name="textBox">TextBox string.</param>
        /// <param name="lblText">Text for lblText</param>
        public OpenFOAMTextBoxForm(Regex reg, string lblTextBox, string lblVariable)
        {
            InitializeComponent();
            InitializeTextBox(lblTextBox);
            m_RegTxt = reg;
            SetLBLVariable(lblVariable);
        }
        #endregion

        /// <summary>
        /// Initialize Textbox.
        /// </summary>
        /// <param name="text">string for text</param>
        private void InitializeTextBox(string text)
        {
            m_TxtBox = textBox1;
            m_TxtBox.Text = text;
        }

        /// <summary>
        /// Set the text for lblText.
        /// </summary>
        /// <param name="txt">string for text.</param>
        public void SetLBLText(string txt)
        {
            lblTxt.Text = txt;
        }

        /// <summary>
        /// Set the text for lblVaribal
        /// </summary>
        /// <param name="txt">string for text</param>
        public void SetLBLVariable(string txt)
        {
            lblEnvironmentVariable.Text = txt;
        }

        /// <summary>
        /// Getter-Setter for textBox.
        /// </summary>
        public TextBox TxtBox
        {
            get
            {
                return m_TxtBox;
            }
            set
            {
                m_TxtBox = value;
            }
        }

        /// <summary>
        /// Getter for regular expression for textbox.
        /// </summary>
        public Regex RegText
        {
            get
            {
                return m_RegTxt;
            }
        }

        /// <summary>
        /// Getter-Setter for boolean m_CancelProcess.
        /// </summary>
        public bool CancelProcess
        {
            get { return m_CancelProcess; }
            set { m_CancelProcess = value; }
        }

        /// <summary>
        /// Cancel button click event.
        /// </summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The event args.</param>
        private void BtnCancel_Click(object sender, EventArgs e)
        {
            m_CancelProcess = !m_CancelProcess;

            if (m_CancelProcess)
                Close();
        }

        /// <summary>
        /// Click-Event for save button.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BtnSave_Click(object sender, EventArgs e)
        {
            //TO-DO: NEEDS TO BE REWORKED TO SAVE VALUE OF TXTBOX IN OWN ATTRIBUTE INTO THIS CLASS.
            Close();
        }

        /// <summary>
        /// Click-Event for help button.
        /// </summary>
        /// <param name="sender">Sender object.</param>
        /// <param name="e">Event args.</param>
        private void BtnHelp_Click(object sender, EventArgs e)
        {
            //TO-DO: REWORK WITH OPENFOAMEXPORTRESOURCE.
            MessageBox.Show("Please insert the variable path that is listed on the left.", OpenFOAMInterfaceResource.MESSAGE_BOX_TITLE);
        }
    }
}
