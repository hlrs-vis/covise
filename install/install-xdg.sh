xdg-mime install hlrs-net.xml

xdg-desktop-menu install --noupdate hlrs-covise.desktop
xdg-desktop-menu install --noupdate hlrs-cover.desktop
xdg-desktop-menu install --noupdate hlrs-tablet.desktop
xdg-desktop-menu forceupdate

xdg-icon-resource install --noupdate --size 48 hlrs-cover.png
xdg-icon-resource install --noupdate --size 48 hlrs-covise.png
xdg-icon-resource install --noupdate --context mimetypes --size 48 hlrs-covise.png text-x-net
xdg-icon-resource forceupdate
