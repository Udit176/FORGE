#!/usr/bin/env python3
"""
Test runner for all unit tests.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py -v  # Verbose
    python tests/run_tests.py --quick  # Quick subset
"""

import unittest
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests(verbose: int = 1):
    """Run all unit tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_quick_tests(verbose: int = 1):
    """Run a quick subset of tests."""
    loader = unittest.TestLoader()

    # Quick test suite
    suite = unittest.TestSuite()

    # Add essential tests
    from tests.test_environment import TestObstacle, TestDrivingEnv
    from tests.test_oracle import TestAStarPlanner
    from tests.test_teacher import TestDemonstratorParams, TestSyntheticTeacher
    from tests.test_methods import TestMLPPolicy

    suite.addTests(loader.loadTestsFromTestCase(TestObstacle))
    suite.addTests(loader.loadTestsFromTestCase(TestDrivingEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestAStarPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestDemonstratorParams))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticTeacher))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPPolicy))

    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Run unit tests')
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Increase verbosity')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset of tests')
    args = parser.parse_args()

    print("=" * 60)
    print("Forward-Model Guided Demonstration Learning - Unit Tests")
    print("=" * 60)

    if args.quick:
        print("\nRunning quick test suite...\n")
        success = run_quick_tests(args.verbose)
    else:
        print("\nRunning all tests...\n")
        success = run_all_tests(args.verbose)

    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
