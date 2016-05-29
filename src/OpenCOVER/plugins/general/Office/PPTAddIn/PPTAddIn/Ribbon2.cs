using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Office.Tools.Ribbon;

namespace PPTAddIn
{
    public partial class Ribbon2
    {

        private void port_Click(object sender, RibbonControlEventArgs e)
        {
            host.Text = "visent.hlrs.de";
            port.Text = "31315";
        }
        private void Ribbon2_Load(object sender, RibbonUIEventArgs e)
        {

        }

        private void connectToCAVE_Click(object sender, RibbonControlEventArgs e)
        {
            host.Text = "visent.hlrs.de";
            port.Text = "31315";
        }
    }
}
