# Draft Strategy

## Core principles
- Bench players never score in this format: judge every pick by how it upgrades the
  starting 9 (QB, 2 RB, 2 WR, TE, FLEX, K, DST).
- Value = projected points above the replacement-level starter still available at
  the position when you next pick. Scarcity at RB/WR matters more than raw points.
- Use ADP as the market price. Take players falling meaningfully below ADP; avoid
  reaching more than a round early without a news-based reason.

## Round shape (adjust to how the draft breaks)
- Rounds 1-5: lock in RB/WR starters; take an elite TE or QB only at a clear discount.
- Rounds 6-9: fill remaining starters (QB, TE, FLEX) targeting upside profiles.
- Rounds 10-13: bench = insurance ONLY for fragile starters; prefer upgrading
  starter certainty over depth.
- Last 2 rounds: K and DST, best available by ADP.

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
