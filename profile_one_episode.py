"""Profile one PPS RL episode to find the real bottleneck."""
import cProfile
import pstats
from pps_env import PPSEnv

env = PPSEnv(max_episode_ticks=3000)

def run_one():
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

profiler = cProfile.Profile()
profiler.enable()
run_one()
profiler.disable()

# Save full profile
profiler.dump_stats("pps_episode.prof")

# Print top 30 hot functions by cumulative time
print("\n=== TOP 30 BY CUMULATIVE TIME ===")
stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(30)

print("\n=== TOP 30 BY SELF TIME ===")
stats = pstats.Stats(profiler).sort_stats("tottime")
stats.print_stats(30)
