"""Explicit random-ripple mode of full native held-out and real-voyage audit."""
import sys
from EvaluateNoFIRSonarTrial import main

if __name__ == '__main__':
    sys.argv.append('--random-ripple')
    main()
