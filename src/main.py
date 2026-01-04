#!/usr/bin/env python3
"""
Fox-2 Missile Path Probabilistic Visualizer

Main entry point for running missile-target engagement simulations.

Usage:
    python main.py --preset AIM-9L --runs 100
    python main.py --preset AIM-9B --evasion random --flares 60
    python main.py --range 5 --aspect 30 --runs 50
"""

import simulation.visual as visual
from entities.target import EvasiveManeuver
from simulation.simulation import Simulation, print_results
from core.config import MISSILE_PRESETS, TARGET_PRESETS
import argparse
import sys
import time
import os
from typing import Optional

# Add the src directory to Python path so submodule imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fox-2 Missile Path Probabilistic Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --preset AIM-9L --runs 100
  python main.py --preset AIM-9B --evasion random --flares 60
  python main.py --range 5 --aspect 180 --runs 50
  
Missile Presets:
  AIM-9B  - 1956, rear-aspect only, no IRCCM
  AIM-9D  - 1965, improved seeker, 18G
  AIM-9G  - 1970, SEAM capable, 18G
  AIM-9L  - 1977, all-aspect, 35G
  AIM-9M  - 1982, improved IRCCM, 35G
  AIM-9X  - 2003, PLACEHOLDER values

Evasion Modes:
  none        - No evasion
  random      - Random maneuvers
  break_left  - Hard left turn
  break_right - Hard right turn
  notch       - Beam (perpendicular to threat)
        """
    )

    # Missile configuration
    parser.add_argument(
        '--preset', '-p',
        choices=list(MISSILE_PRESETS.keys()),
        default='AIM-9L',
        help='Missile preset (default: AIM-9L)'
    )

    # Target configuration
    parser.add_argument(
        '--target', '-t',
        choices=list(TARGET_PRESETS.keys()),
        default='fighter',
        help='Target type (default: fighter)'
    )

    # Engagement parameters
    parser.add_argument(
        '--range', '-r',
        type=float,
        default=5.0,
        help='Initial range in km (default: 5.0)'
    )

    parser.add_argument(
        '--aspect', '-a',
        type=float,
        default=0.0,
        help='Aspect angle in degrees, 0=tail, 180=head-on (default: 0)'
    )

    parser.add_argument(
        '--altitude',
        type=float,
        default=5000.0,
        help='Engagement altitude in meters (default: 5000)'
    )

    # Evasion
    parser.add_argument(
        '--evasion', '-e',
        choices=['none', 'random', 'break_left', 'break_right', 'notch'],
        default='none',
        help='Target evasion mode (default: none)'
    )

    parser.add_argument(
        '--flares', '-f',
        type=int,
        default=60,
        help='Number of flares on target (default: 60)'
    )

    # Simulation parameters
    parser.add_argument(
        '--runs', '-n',
        type=int,
        default=100,
        help='Number of Monte Carlo runs (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    # Output options
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization (just print results)'
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save plot to file (e.g., results.png)'
    )

    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create animated replay (shows first run)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison across all missile presets'
    )

    return parser.parse_args()


def progress_bar(current: int, total: int, width: int = 40):
    """Print a progress bar."""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r[{bar}] {current}/{total} ({progress:.0%})', end='', flush=True)


def run_single_scenario(args) -> None:
    """Run a single simulation scenario."""
    print(f"\n{'='*60}")
    print(f"Fox-2 Missile Engagement Simulation")
    print(f"{'='*60}")
    print(f"Missile:   {args.preset}")
    print(f"Target:    {args.target}")
    print(f"Range:     {args.range} km")
    print(f"Aspect:    {args.aspect}° (0=tail, 180=head-on)")
    print(f"Evasion:   {args.evasion}")
    print(f"Flares:    {args.flares}")
    print(f"Runs:      {args.runs}")
    print(f"{'='*60}")

    # Create simulation
    sim = Simulation.create_engagement_scenario(
        missile_preset=args.preset,
        target_preset=args.target,
        evasion=args.evasion,
        initial_range_km=args.range,
        aspect_angle_deg=args.aspect,
        altitude_m=args.altitude,
        flare_count=args.flares,
        num_runs=args.runs
    )

    # Set seed if provided
    if args.seed is not None:
        sim.sim_config.random_seed = args.seed

    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()

    def on_progress(current, total):
        progress_bar(current, total)

    results = sim.run_monte_carlo(progress_callback=on_progress)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")

    # Print results
    print_results(results)

    # Visualization
    if not args.no_plot:
        print("\nGenerating visualization...")

        fig = visual.plot_monte_carlo_results(results)

        if args.save:
            visual.save_figure(fig, args.save)

        if args.animate and len(results.individual_results) > 0:
            print("Creating animation...")
            anim = visual.create_animation(results, run_index=0)

        visual.show()


def run_comparison(args) -> None:
    """Run comparison across all missile presets."""
    print("\n" + "="*60)
    print("Missile Comparison Study")
    print("="*60)
    print(f"Range:     {args.range} km")
    print(f"Evasion:   {args.evasion}")
    print(f"Flares:    {args.flares}")
    print(f"Runs/missile: {args.runs}")
    print("="*60)

    missile_names = list(MISSILE_PRESETS.keys())
    pk_values = []

    for preset in missile_names:
        print(f"\nTesting {preset}...")

        sim = Simulation.create_engagement_scenario(
            missile_preset=preset,
            target_preset=args.target,
            evasion=args.evasion,
            initial_range_km=args.range,
            aspect_angle_deg=args.aspect,
            altitude_m=args.altitude,
            flare_count=args.flares,
            num_runs=args.runs
        )

        results = sim.run_monte_carlo(
            progress_callback=lambda c, t: progress_bar(c, t, width=30)
        )
        print()

        pk_values.append(results.pk)
        print(f"  Pk = {results.pk:.1%}")

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    for name, pk in zip(missile_names, pk_values):
        bar_len = int(pk * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"{name:8s} [{bar}] {pk:5.1%}")
    print("="*60)

    # Plot comparison
    if not args.no_plot:
        fig = visual.plot_outcome_comparison(missile_names, pk_values)
        if args.save:
            visual.save_figure(fig, args.save)
        visual.show()


def run_interactive() -> None:
    """Run interactive mode for exploring scenarios."""
    print("\n" + "="*60)
    print("Interactive Fox-2 Missile Simulator")
    print("="*60)

    while True:
        print("\nOptions:")
        print("  1. Run single simulation")
        print("  2. Compare all missiles")
        print("  3. Show missile presets")
        print("  4. Quit")

        choice = input("\nChoice [1-4]: ").strip()

        if choice == '1':
            preset = input(
                "Missile preset (AIM-9B/D/G/L/M/X) [AIM-9L]: ").strip() or "AIM-9L"
            target = input(
                "Target type (fighter/transport) [fighter]: ").strip() or "fighter"
            range_km = float(input("Range in km [5]: ").strip() or "5")
            aspect = float(input(
                "Aspect angle (0=tail, 90=beam, 180=head-on) [0]: ").strip() or "0")
            altitude = float(
                input("Altitude in meters [5000]: ").strip() or "5000")
            evasion = input(
                "Evasion (none/random/break_left/break_right/notch) [none]: ").strip() or "none"
            flares = int(input("Number of flares [60]: ").strip() or "60")
            runs = int(input("Number of runs [50]: ").strip() or "50")

            args = argparse.Namespace(
                preset=preset,
                target=target,
                range=range_km,
                aspect=aspect,
                altitude=altitude,
                evasion=evasion,
                flares=flares,
                runs=runs,
                seed=None,
                no_plot=False,
                save=None,
                animate=False
            )
            run_single_scenario(args)

        elif choice == '2':
            runs = int(input("Runs per missile [30]: ").strip() or "30")
            args = argparse.Namespace(
                preset='AIM-9L',
                target='fighter',
                range=5.0,
                aspect=0.0,
                altitude=5000.0,
                evasion='none',
                flares=60,
                runs=runs,
                seed=None,
                no_plot=False,
                save=None
            )
            run_comparison(args)

        elif choice == '3':
            print("\nMissile Presets:")
            print("-" * 50)
            for name, config in MISSILE_PRESETS.items():
                print(f"\n{name}:")
                print(f"  Max G: {config.max_g}")
                print(f"  Range: {config.max_range_km} km")
                print(f"  Seeker: {config.seeker_type.value}")
                print(f"  IRCCM: {config.irccm_level.value}")
                print(f"  Source: {config.source_notes}")

        elif choice == '4':
            print("Goodbye!")
            break


def main():
    """Main entry point."""
    args = parse_args()

    # Check for interactive mode (no arguments)
    if len(sys.argv) == 1:
        run_interactive()
        return

    if args.compare:
        run_comparison(args)
    else:
        run_single_scenario(args)


if __name__ == '__main__':
    main()
