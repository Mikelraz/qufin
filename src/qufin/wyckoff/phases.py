"""
Rule-based Wyckoff phase (A-E) classification within a trading range.

Given the events found by :mod:`qufin.wyckoff.events`, this module walks them
in chronological order per trading range and assigns Phase A through E.

Accumulation
------------
* **Phase A** — Stopping action: PS → SC → AR → ST.
* **Phase B** — Building the cause: oscillation between AR-high and ST-low.
* **Phase C** — Test of supply: Spring (or final ST) below support.
* **Phase D** — Markup developing: dominant SOS/LPS sequence.
* **Phase E** — Range exit confirmed: clean break above resistance.

Distribution mirrors the above with PSY/BC/AR/ST/UT/UTAD/SOW/LPSY.
"""

from __future__ import annotations

import numpy as np

from ._types import (
    OHLCV,
    ClimaxEvent,
    SpringEvent,
    StructuralEvent,
    TradingRange,
    WyckoffPhase,
)


def classify_phases(
    bars: OHLCV,
    ranges: list[TradingRange],
    climaxes: list[ClimaxEvent],
    structural: list[StructuralEvent],
    springs: list[SpringEvent],
    upthrusts: list[SpringEvent] | None = None,
) -> list[WyckoffPhase]:
    """
    Assign Wyckoff phases A-E to each trading range.

    Parameters
    ----------
    bars        OHLCV sequence (for range-exit detection).
    ranges      Detected trading ranges, in chronological order.
    climaxes    SC / BC events from ``detect_climax``.
    structural  PS / AR / ST / SOS / LPS / SOW / LPSY events.
    springs     Spring events from ``detect_spring``.
    upthrusts   UT / UTAD events from ``detect_upthrust``; may be None.

    Returns
    -------
    list of WyckoffPhase ordered by ``start_idx``.
    """
    upthrusts = upthrusts or []
    out: list[WyckoffPhase] = []
    close = bars.close()

    for tr in ranges:
        tr_climaxes = [c for c in climaxes if tr.start_idx <= c.idx < tr.end_idx]
        if not tr_climaxes:
            continue
        # Use the first climax to determine schematic direction.
        schematic = "Acc" if tr_climaxes[0].kind == "SC" else "Dist"

        tr_struct = [s for s in structural if tr.start_idx <= s.idx < tr.end_idx]
        tr_springs = [s for s in springs if tr.start_idx <= s.idx < tr.end_idx]
        tr_upthrusts = [u for u in upthrusts if tr.start_idx <= u.idx < tr.end_idx]
        post_struct = [s for s in structural if s.idx >= tr.end_idx]

        # ----- Phase A: stopping action — start of range to AR or first ST -----
        sc = tr_climaxes[0]
        ar = next((s for s in tr_struct if s.kind == "AR"), None)
        st = next((s for s in tr_struct if s.kind == "ST"), None)
        phase_a_end = (st.idx + 1) if st is not None else (ar.idx + 1 if ar is not None else None)
        if phase_a_end is None:
            continue
        out.append(
            WyckoffPhase(
                start_idx=int(tr.start_idx),
                end_idx=int(phase_a_end),
                schematic=schematic,  # type: ignore[arg-type]
                phase="A",
            )
        )

        # ----- Phase C: spring (acc) or UTAD (dist) -----
        c_event_idx: int | None = None
        if schematic == "Acc" and tr_springs:
            c_event_idx = max(s.idx for s in tr_springs)
        elif schematic == "Dist" and tr_upthrusts:
            c_event_idx = max(u.idx for u in tr_upthrusts)

        # ----- Phase B: between end of A and start of C -----
        phase_b_end = c_event_idx if c_event_idx is not None else tr.end_idx
        if phase_b_end > phase_a_end:
            out.append(
                WyckoffPhase(
                    start_idx=int(phase_a_end),
                    end_idx=int(phase_b_end),
                    schematic=schematic,  # type: ignore[arg-type]
                    phase="B",
                )
            )

        # ----- Phase C: a small window around the spring/upthrust -----
        if c_event_idx is not None:
            c_end = min(int(tr.end_idx), int(c_event_idx) + 1)
            if c_end > c_event_idx:
                out.append(
                    WyckoffPhase(
                        start_idx=int(c_event_idx),
                        end_idx=int(c_end),
                        schematic=schematic,  # type: ignore[arg-type]
                        phase="C",
                    )
                )

        # ----- Phase D: post-range SOS/LPS (acc) or SOW/LPSY (dist) -----
        sos_kind = "SOS" if schematic == "Acc" else "SOW"
        lps_kind = "LPS" if schematic == "Acc" else "LPSY"
        sos_events = [s for s in post_struct if s.kind == sos_kind]
        lps_events = [s for s in post_struct if s.kind == lps_kind]
        if sos_events:
            d_start = sos_events[0].idx
            if lps_events:
                d_end = lps_events[-1].idx + 1
            else:
                d_end = sos_events[-1].idx + 1
            out.append(
                WyckoffPhase(
                    start_idx=int(d_start),
                    end_idx=int(d_end),
                    schematic=schematic,  # type: ignore[arg-type]
                    phase="D",
                )
            )

            # ----- Phase E: confirmed exit beyond D -----
            n = bars.n_bars
            if d_end < n:
                exit_idx = _confirm_range_exit(close, d_end, tr, schematic)
                if exit_idx is not None and exit_idx < n:
                    out.append(
                        WyckoffPhase(
                            start_idx=int(exit_idx),
                            end_idx=int(n),
                            schematic=schematic,  # type: ignore[arg-type]
                            phase="E",
                        )
                    )

        _ = sc  # SC used only to set schematic; kept named for readability.

    return out


def _confirm_range_exit(
    close: np.ndarray, start_idx: int, tr: TradingRange, schematic: str
) -> int | None:
    """First bar from ``start_idx`` whose close exits the range definitively."""
    n = close.shape[0]
    for i in range(start_idx, n):
        if schematic == "Acc" and close[i] > tr.resistance:
            return i
        if schematic == "Dist" and close[i] < tr.support:
            return i
    return None
