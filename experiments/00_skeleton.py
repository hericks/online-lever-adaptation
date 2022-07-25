# Relative imports outside of package
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner