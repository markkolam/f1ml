# Pirelli compound nominations

Timing data only labels tyres SOFT / MEDIUM / HARD. What a compound *is*
changes from weekend to weekend (the C3 can be the Hard, Medium or Soft) and
from season to season (Buscombe: the C3 went from 0.27 s/lap faster than the C2
in 2023-25 to 0.44 s/lap slower in 2026). Models therefore key compounds on the
Pirelli C-number, not the label.

Add one file per season, `<season>.yaml`, keyed by round number:

```yaml
season: 2026
nominations:
  1: {HARD: C1, MEDIUM: C2, SOFT: C3}
  2: {HARD: C2, MEDIUM: C3, SOFT: C4}
```

Source: Pirelli's published nominations per round. Rounds without an entry get
`c_number = null` in the clean laps table and are reported by `f1ml status`.
