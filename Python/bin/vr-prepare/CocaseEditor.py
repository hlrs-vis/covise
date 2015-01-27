
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import os
import os.path
import sys

if __name__ == "__main__":
    coviseDir = os.getenv('COVISEDIR')
    if not coviseDir:
        print ('Program stop!  Environment variable COVISEDIR is not set!'
               '  COVISEDIR is needed to point to a covise location.')
        sys.exit()
    sys.path.append(os.path.join(coviseDir, 'src/application/ui/vr-prepare'))

from CocaseEditorMain import main


main()

# eof
