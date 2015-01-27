proc buttonConfigMAP { hostname number button } {
    global ListForSection
    set ListForSection(ButtonConfig,$hostname,$number) $button
}



proc saveButtonConfig { body hostname } {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection(ButtonConfig,SERIAL_PORT,$hostname)] {
        set b "$b     SERIAL_PORT $ListForSection(ButtonConfig,SERIAL_PORT,$hostname)\n"
    }
    for { set i 1 } { $i < 5 } { incr i } {
        if [canBeSaved ListForSection(ButtonConfig,$i,$hostname)] {
            set b "$b      MAP $i $ListForSection(ButtonConfig,$i,$hostname)\n"
        }
    }
}


proc makeButtonConfigGUI { hostname pk }  {
    global ListForSection
    set f [getFrame ButtonConfig $pk]

    label $f.lserial -text "Serial port"
    bind  $f.lserial <Button-3> "showHelp ButtonConfig SERIAL_PORT" 
    entry $f.eserial -textvariable ListForSection(ButtonConfig,SERIAL_PORT,$hostname) -width 12
    grid $f.lserial  -row 0 -column 0 
    grid $f.eserial -row 0 -column 1
    
    label $f.lButton -text "Button number"
    grid  $f.lButton -row 1 -column 0

    label $f.lEnum -text "Button Action"
    grid  $f.lEnum -row 1 -column 2

    for { set i 1 } { $i < 5 } { incr i } {
        set currCol 1 
        label $f.l${i} -text $i
        grid  $f.l${i} -row [expr $i+1]  -column 0
        foreach j { ACTION_BUTTON DRIVE_BUTTON XFORM_BUTTON NONE } {
            radiobutton $f.rb${j}${i} -text $j -variable ListForSection\(ButtonConfig,$i,$hostname\) -value $j
            bind $f.rb${j}${i}  <Button-3> "showHelp ButtonConfig MAP"
            grid $f.rb${j}${i} -row [expr $i+1] -column $currCol
            incr currCol
        }
    }
}




#####End of buttonconfig.tcl###################################################
###############################################################################

