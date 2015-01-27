proc SCREENCONFIGITEM { screenNo name sizeX sizeY origX origY origZ head pitch roll hostname} { 
    global ListForSection
    set ListForSection(ScreenConfig,name,$screenNo,$hostname) $name
    set ListForSection(ScreenConfig,sizeX,$screenNo,$hostname) $sizeX
    set ListForSection(ScreenConfig,sizeY,$screenNo,$hostname) $sizeY
    set ListForSection(ScreenConfig,origX,$screenNo,$hostname) $origX
    set ListForSection(ScreenConfig,origY,$screenNo,$hostname) $origY
    set ListForSection(ScreenConfig,origZ,$screenNo,$hostname) $origZ
    set ListForSection(ScreenConfig,head,$screenNo,$hostname) $head
    set ListForSection(ScreenConfig,pitch,$screenNo,$hostname) $pitch
    set ListForSection(ScreenConfig,roll,$screenNo,$hostname) $roll
}

proc saveScreenConfig { body hostname } {
    upvar $body b
    global ListForSection
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_SCREENS,$hostname) } { incr i } {
        if { [canBeSaved ListForSection(ScreenConfig,name,$i,$hostname)]  &&
             [canBeSaved ListForSection(ScreenConfig,sizeX,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,sizeY,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,origX,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,origY,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,origZ,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,head,$i,$hostname)]  &&
             [canBeSaved ListForSection(ScreenConfig,pitch,$i,$hostname)] &&
             [canBeSaved ListForSection(ScreenConfig,roll,$i,$hostname)]} {
                 set b "$b     $i     \
                         $ListForSection(ScreenConfig,name,$i,$hostname) \
                         $ListForSection(ScreenConfig,sizeX,$i,$hostname) \
                         $ListForSection(ScreenConfig,sizeY,$i,$hostname) \
                         $ListForSection(ScreenConfig,origX,$i,$hostname) \
                         $ListForSection(ScreenConfig,origY,$i,$hostname) \
                         $ListForSection(ScreenConfig,origZ,$i,$hostname) \
                         $ListForSection(ScreenConfig,head,$i,$hostname) \
                         $ListForSection(ScreenConfig,pitch,$i,$hostname) \
                         $ListForSection(ScreenConfig,roll,$i,$hostname)\n"
             }
    }
}

proc makeScreenConfigGUI { hostname pk} {
     global ListForSection
     set f [getFrame ScreenConfig $pk]
     set i 0
     foreach { l t } { number "Screen Number" \
                        name Name \
                        sizeX "Horizontal\n size" \
                        sizeY "Vertical\n size" \
                        origX "X-Origin" \
                        origY "Y-Origin" \
                        origZ "Z-Origin" \
                        head Head \
                        pitch Pitch \
                        roll Roll } {
        label $f.$l -text $t
        bind $f.$l <Button-3> "showHelp ScreenConfig $l"
        grid  $f.$l -row 0 -column $i -sticky w
        incr i
     }
     for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_SCREENS,$hostname) } { incr i } {
        label $f.lScreenNo$i -text $i
        grid  $f.lScreenNo$i -row [expr $i+1] -column 0
        set j 1
        foreach e { name sizeX sizeY origX origY origZ head pitch roll } {
                entry $f.e$e$i -textvariable ListForSection(ScreenConfig,$e,$i,$hostname) -width 10
                bind $f <Button-3> "showHelp ScreenConfig $t"
                grid $f.e$e$i -column $j -row [expr $i+1]
                incr j
        }
     }
}