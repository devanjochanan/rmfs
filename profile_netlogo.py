# profile_netlogo.py
import cProfile
import pstats
import io
import netlogo  # Your own netlogo.py module

def run_netlogo_simulation():
    netlogo.setup()
    # for _ in range(1000):
        # netlogo.tick()
    netlogo.console_tick()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_netlogo_simulation()

    profiler.disable()

    profiler.dump_stats("profile.prof")
    with open("netlogo_profile_summary.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('tottime')
        stats.print_stats(20)

    # Print the top time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())
