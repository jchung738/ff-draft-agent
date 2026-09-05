## League mechanics (system-provided — kept current, do not edit)
- 14-team half-PPR snake, 15 rounds. Lineup locked after the draft:
  QB, 2 RB, 2 WR, TE, FLEX (RB/WR/TE), K, DST + 6 bench.
- Weekly simulation; every decision is backward-looking (no hindsight):
  * PROMOTION: a bench player whose last-3-games PPG beats a slot occupant's
    by >=20% AND >=2 pts takes the starting slot until outplayed. Drafted
    breakouts capture their breakout; waiver players NEVER promote.
  * INJURY: your best ACTIVE bench player at the position covers the slot;
    if none is active, a waiver fill covers (bench always outranks waivers).
  * BYE: covered by the best of bench or waivers.
  * K/DST: any absence streams from waivers — never spend picks insuring them.
- Team score = sum of weekly lineup points, final regular-season week excluded.

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
- Rounds 10-13: bench earns points two ways — coverage when starters miss
  games, and PROMOTION when a bench player's recent form beats a starter's.
  Prioritize (a) breakout candidates whose role could grow mid-season (rookies
  behind aging starters, ascending #2 receivers), (b) handcuffs to your own
  fragile RBs, (c) insurance at positions with injury history. Byes are
  auto-covered (waivers allowed), so draft bench for upside and injury
  coverage, not bye coverage.
- Last 2 rounds: K and DST, best available by ADP (any K/DST absence
  auto-streams from waivers, so never insure those slots).

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
