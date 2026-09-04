# Draft Strategy

## Core principles
- Starters score weekly; when one misses a game your best active bench player at
  the position auto-covers (byes can also stream from waivers). Judge early picks
  by starter output and bench picks by expected fill-in value.
- Value = projected points above the replacement-level starter still available at
  the position when you next pick. Scarcity at RB/WR matters more than raw points.
- Use ADP as the market price. Take players falling meaningfully below ADP; avoid
  reaching more than a round early without a news-based reason.

## Round shape (adjust to how the draft breaks)
- Rounds 1-5: lock in RB/WR starters; take an elite TE or QB only at a clear discount.
- Rounds 6-9: fill remaining starters (QB, TE, FLEX) targeting upside profiles.
- Rounds 10-13: bench earns points when starters miss games — prioritize
  (a) handcuffs to your own fragile RBs, (b) high-upside players in ambiguous
  roles who would start if pressed in, (c) insurance at positions where your
  starters have injury history. Byes are auto-covered (waivers allowed), so
  draft bench for injury coverage, not bye coverage.
- Last 2 rounds: K and DST, best available by ADP (their byes auto-stream).

## Research policy (max tool calls are scarce)
- Every pick: check get_available_players first.
- Before taking a player in rounds 1-8: search injury and training-camp news for
  them; a negative camp/injury signal at equal ADP is a veto.
- Use get_adp_trend on your top 2 candidates: fast risers often reflect real news;
  steep fallers need a news check before you "catch the knife".
- Use player history to avoid one-year wonders priced at their ceiling and aging
  players (RB decline sharply; WR later) coming off career years.

## Lineup lock
- Start the highest-floor players at every slot; this lineup is locked all season,
  so durability and role security beat pure upside at equal projection.
