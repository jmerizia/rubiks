import pstats
p = pstats.Stats('stats')
p.sort_stats('time').print_stats(1000)
