"""Textual rewrite of the Perferox HTML dashboard mock."""

from __future__ import annotations

import os
from typing import ClassVar

# Textual reads color env vars at import time; keep the HTML Gruvbox hexes intact.
os.environ.pop("NO_COLOR", None)
os.environ.setdefault("TEXTUAL_COLOR_SYSTEM", "truecolor")

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.widgets import Button, Input, Static

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

SUBAGENTS = (
  ("sa-01", "A100-80G", 4, 5, "12m"),
  ("sa-02", "MI300X", 1, 6, "41m"),
  ("sa-03", "H100", 3, 5, "1h02"),
  ("sa-04", "L40S", 1, 4, "00m"),
  ("sa-05", "MI250", 2, 4, "22m"),
)

TRACE_ROWS = (
  ("◈", "#d3869b", "THINKING", "#b16286", "output_tps variance on MI300X exceeds 15% between adjacent commits — bisecting the FlashInfer decode path first.", "#bdae93"),
  ("›", "#8ec07c", "RUN_EXPERIMENT", "#8ec07c", "{ commit: a3f19c, backend: flashinfer, chip: MI300X, bs: 64, dtype: fp8 }", "#7c6f64"),
  ("", "#b8bb26", "RESULT", "#504945", "output_tps 812→503 (−38%) · ttft_p50 41ms · cache_hit 0.71 · peak_mem 74.8G", "#b8bb26"),
  ("⚠", "#fb4934", "ANOMALY", "#fb4934", "flagged ANM-014 · output_tps regression on MI300X — deduped vs 3 prior runs, novel.", "#ea6962"),
  ("◈", "#d3869b", "THINKING", "#b16286", "dedup: hash(commit,backend,chip,bs,dtype) → 0x9f2a4c not in ledger. proceeding, no retry.", "#bdae93"),
  ("›", "#8ec07c", "SPAWN_SUBAGENT", "#8ec07c", "{ chip: L40S, task: backend-matrix, budget: 25 runs }", "#7c6f64"),
  ("", "#b8bb26", "RESULT", "#504945", "sa-04 booting on L40S … image pull 2.1GB", "#b8bb26"),
  ("◈", "#d3869b", "THINKING", "#b16286", "H100 parity holds (cos_sim 0.9997). shifting pressure to the ill-maintained triton backend.", "#bdae93"),
  ("›", "#8ec07c", "RUN_EXPERIMENT", "#8ec07c", "{ commit: main, backend: triton, chip: A100, bs: 128, chunked_prefill: off }", "#7c6f64"),
  ("", "#b8bb26", "RESULT", "#504945", "ttft_p99 512ms (+410ms) · tpot_p99 38.7ms · error_rate 0.4% · p50 stable", "#b8bb26"),
  ("⚠", "#fb4934", "ANOMALY", "#fb4934", "flagged ANM-011 · ttft_p99 tail blowup on A100 under triton backend.", "#ea6962"),
  ("·", "#504945", "LEDGER", "#504945", "214 experiments · 6 anomalies · retries avoided this cycle: 37", "#6b7684"),
  ("◈", "#d3869b", "THINKING", "#b16286", "cosine drift on H100 fp8 vs fp16 reference at 0.982 — below the 0.99 parity gate.", "#bdae93"),
  ("›", "#8ec07c", "WRITE_RESULT", "#8ec07c", "{ run: 0142, table: experiments, rows: +18 }", "#7c6f64"),
  ("", "#b8bb26", "RESULT", "#504945", "committed · sqlite experiments=214 anomalies=6 runs=1.4k", "#b8bb26"),
)

ANOMALIES = (
  {
    "aid": "ANM-014",
    "title": "output_tps regression",
    "sub": "sa-02",
    "chip": "MI300X",
    "commit": "a3f19c",
    "ts": "14:22:07Z",
    "metric": "output_tps 812→503  −38%",
    "metric_color": "#fb4934",
    "detail": "output_tps on MI300X fell from 812 to 503 (−38%) between a3f19c and its parent, isolated to the FlashInfer decode path under fp8 with bs=64. TTFT and cache-hit held steady, so this is pure decode throughput, not scheduling. Reproduced 3/3 on identical inputs; cosine similarity vs the fp16 reference stayed at 0.9997 — outputs correct, only speed regressed. Suspect a kernel-selection change in the fp8 GEMM autotuner. Novel: deduped against 3 adjacent runs.",
  },
  {
    "aid": "ANM-013",
    "title": "cache-hit collapse",
    "sub": "sa-05",
    "chip": "MI250",
    "commit": "7c1e04",
    "ts": "13:58:41Z",
    "metric": "cache_hit 0.71→0.50  −0.21",
    "metric_color": "#fb4934",
    "detail": "cache_hit_rate on MI250 drifted from 0.71 to 0.50 across a run of radix-cache commits. The drop tracks a change to prefix-matching eviction; longer shared prefixes stopped hitting after ~2k requests. Throughput held but effective KV reuse collapsed, inflating cost per request. Not an accuracy issue — cosine unchanged. Flagged for a focused bisect over the radix-cache range; two earlier near-duplicates were merged into this entry.",
  },
  {
    "aid": "ANM-011",
    "title": "ttft p99 blowup",
    "sub": "sa-01",
    "chip": "A100-80G",
    "commit": "b920af",
    "ts": "13:41:12Z",
    "metric": "ttft_p99 +410ms",
    "metric_color": "#fe8019",
    "detail": "ttft_p99 on A100 spiked to 512ms (+410ms) under the triton backend at bs=128, while p50 stayed at 41ms — a tail, not a mean, regression. Correlates with scheduler queueing when a large prefill lands mid-batch; the triton path lacks the chunked-prefill guard the default backend has. error_rate rose to 0.4% from timeouts. Outputs correct. Recommend re-running with chunked prefill forced on to confirm the guard is the cause.",
  },
  {
    "aid": "ANM-009",
    "title": "cosine drift",
    "sub": "sa-03",
    "chip": "H100",
    "commit": "4de77a",
    "ts": "12:57:03Z",
    "metric": "cos_sim 0.982 (< 0.99)",
    "metric_color": "#fb4934",
    "detail": "cosine similarity on H100 fp8 vs the fp16 reference dropped to 0.982, below the 0.99 parity gate — a correctness anomaly, not perf. Divergence concentrates in long-context outputs past ~4k tokens, consistent with fp8 KV-cache accumulation error rather than a kernel bug. Reproduced 4/4. Highest-severity finding this cycle: fp8 outputs silently diverge under load. Escalated; stored with full token-level diff in the anomalies table.",
  },
  {
    "aid": "ANM-007",
    "title": "oom at warmup",
    "sub": "sa-06",
    "chip": "H200",
    "commit": "1aa3c2",
    "ts": "12:30:55Z",
    "metric": "peak_mem 79.4G  OOM",
    "metric_color": "#fb4934",
    "detail": "peak_gpu_mem_gb hit 79.4G on H200 during warmup and OOM-killed the server before serving traffic. Triggered by a warmup batch sized from max_num_seqs × max_len without accounting for the CUDA-graph capture buffers on this image. sa-06 is in an error state and was not retried — the config is deterministic and will reproduce. Fix likely a warmup-batch cap or disabling graph capture on constrained memory.",
  },
  {
    "aid": "ANM-004",
    "title": "startup stall",
    "sub": "sa-04",
    "chip": "L40S",
    "commit": "55f0be",
    "ts": "11:48:19Z",
    "metric": "startup +22s",
    "metric_color": "#fe8019",
    "detail": "startup_s on L40S rose by ~22s versus the fleet baseline, isolated to model-weight load, not compilation. The L40S image pulls weights over a slower network mount; warmup itself was nominal. Low severity — no perf or accuracy impact once running — but it inflates cost per experiment and slows the fuzz loop on this chip. Logged so the dedup ledger can down-rank further L40S cold-start runs unless requested.",
  },
)


class AnomalyCard(Static, can_focus=True):
  """Clickable anomaly card from the right-side HTML list."""

  def __init__(self, anomaly: dict[str, str]) -> None:
    """Store one anomaly and render its initial inactive card."""
    self.anomaly = anomaly
    super().__init__("", id=f"card-{anomaly['aid'].lower()}", classes="anomaly-card")
    self.render_card(active=False)

  def render_card(self, *, active: bool) -> None:
    """Refresh text and active-border styling for this card."""
    anomaly = self.anomaly
    self.set_class(active, "active")
    self.update(
      "\n".join(
        (
          f"[#fb4934]●[/] [#ebdbb2]{anomaly['aid']}[/]",
          f"[#bdae93]{anomaly['title']}[/]",
          f"[#7c6f64]{anomaly['sub']}[/] · [#8ec07c]{anomaly['chip']}[/]",
          f"[{anomaly['metric_color']}]{anomaly['metric']}[/]",
        )
      )
    )

  def on_click(self) -> None:
    """Open this anomaly in the split detail pane."""
    self.app.select_anomaly(self.anomaly["aid"])

  def on_key(self, event: Key) -> None:
    """Allow keyboard activation for the focused card."""
    if event.key in {"enter", "space"}:
      self.app.select_anomaly(self.anomaly["aid"])
      event.stop()


class PerferoxTUI(App[None]):
  """Render a Textual version of the HTML dashboard screenshot."""

  CSS = """
  Screen {
    background: #1d2021;
    color: #d5c4a1;
  }

  #root {
    height: 100%;
    width: 100%;
    background: #1d2021;
    layers: base overlay;
  }

  #body {
    height: 1fr;
    background: #1d2021;
    layer: base;
  }

  #subagents {
    width: 23;
    border-right: solid #504945;
  }

  #main {
    width: 1fr;
    min-width: 90;
    border-right: solid #504945;
  }

  #main-split {
    height: 1fr;
    background: #1d2021;
  }

  #trace-pane {
    width: 1fr;
    min-width: 44;
  }

  #detail-pane {
    width: 1fr;
    min-width: 48;
    background: #282828;
    border-left: solid #504945;
  }

  #anomalies {
    width: 27;
    background: #1d2021;
  }

  .section-title {
    height: 2;
    padding: 0 1;
    background: #1d2021;
    color: #fabd2f;
    text-style: bold;
    border-bottom: solid #32302f;
    content-align: left middle;
  }

  .section-title.alert {
    color: #fb4934;
  }

  .scroll-pane {
    height: 1fr;
    padding: 1;
    scrollbar-background: #1d2021;
    scrollbar-background-hover: #1d2021;
    scrollbar-background-active: #1d2021;
    scrollbar-color: #504945;
    scrollbar-color-hover: #504945;
    scrollbar-color-active: #504945;
    scrollbar-corner-color: #1d2021;
  }

  .subagent-card {
    height: 5;
    padding: 0 1;
    margin-bottom: 1;
    background: #282828;
    border: solid #3c3836;
  }

  .trace-line {
    height: auto;
    margin-bottom: 1;
    padding: 0 2;
    background: #1d2021;
  }

  .trace-line.anomaly {
    background: #282828;
    border-left: solid #fb4934;
  }

  .anomaly-card {
    height: 6;
    padding: 0 1;
    margin-bottom: 1;
    background: #282828;
    border: solid #3c3836;
  }

  .anomaly-card.active {
    border: solid #b57614;
    background: #32302f;
  }

  .anomaly-card:focus {
    border: solid #fabd2f;
  }

  #detail-header {
    height: 2;
    padding: 0 1;
    background: #282828;
    color: #fb4934;
    text-style: bold;
    border-bottom: solid #32302f;
    content-align: left middle;
  }

  #detail-header-text {
    width: 1fr;
    height: 1;
  }

  #detail-title {
    height: 3;
    padding: 1 2 0 2;
    color: #ebdbb2;
    text-style: bold;
  }

  #detail-meta {
    height: 3;
    margin: 0 2;
    color: #7c6f64;
    border-top: solid #3c3836;
    border-bottom: solid #3c3836;
  }

  #detail-body {
    height: 1fr;
    padding: 1 2;
    color: #d5c4a1;
  }

  #detail-footer {
    height: 2;
    margin: 0 2;
    padding-top: 1;
    color: #504945;
    border-top: solid #3c3836;
  }

  #prompt-row {
    height: 6;
    padding: 1;
    background: #282828;
    border-top: solid #504945;
  }

  #prompt-icon {
    width: 2;
    height: 3;
    color: #fabd2f;
    content-align: center middle;
  }

  #prompt {
    width: 1fr;
    height: 3;
    margin-right: 1;
    background: #1d2021;
    color: #ebdbb2;
    border: solid #504945;
  }

  #prompt:focus {
    border: solid #b57614;
  }

  #prompt > .input--placeholder {
    color: #7c6f64;
  }

  Button {
    height: 3;
    min-width: 10;
    margin-right: 1;
    background: #32302f;
    color: #fabd2f;
    border: solid #b57614;
    text-style: bold;
    content-align: center middle;
  }

  Button#end {
    color: #7c6f64;
    border: solid #504945;
  }

  Button#close-detail {
    width: 9;
    height: 1;
    min-width: 9;
    margin: 0;
    background: #282828;
    border: none;
    color: #7c6f64;
    text-style: none;
  }

  #footer {
    height: 1;
    padding: 0 1;
    background: #282828;
    color: #7c6f64;
    layer: base;
  }

  #end-confirm {
    display: none;
    layer: overlay;
    position: absolute;
    width: 52;
    height: 12;
    offset: 69 19;
    padding: 1 2;
    background: #282828;
    border: solid #504945;
  }

  #end-confirm-title {
    height: 1;
    color: #fb4934;
    text-style: bold;
  }

  #end-confirm-body {
    height: 2;
    margin-top: 1;
    color: #d5c4a1;
  }

  #end-confirm-actions {
    height: 3;
    margin-top: 1;
  }

  Button#cancel-end {
    color: #fabd2f;
  }

  Button#confirm-end {
    color: #fb4934;
    border: solid #fb4934;
  }
  """

  BINDINGS: ClassVar = [("q", "quit", "Quit"), ("escape", "close_detail", "Close detail")]

  def __init__(self) -> None:
    """Start with ANM-004 open like the screenshot."""
    super().__init__()
    self.anomalies = {anomaly["aid"]: anomaly for anomaly in ANOMALIES}
    self.selected_anomaly = "ANM-004"
    self.spin = 0

  def compose(self) -> ComposeResult:
    """Build the same high-level regions as the HTML mock."""
    with Vertical(id="root"):
      with Horizontal(id="body"):
        yield self._subagents()
        yield self._main()
        yield self._anomalies()
      yield Static("", id="footer")
      yield self._end_overlay()

  def on_mount(self) -> None:
    """Populate the selected anomaly after child widgets exist."""
    self._sync_anomaly_detail()
    self.set_interval(0.12, self._animate_subagents)

  def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle prototype buttons without mutating real benchmark state."""
    button_id = event.button.id
    if button_id == "close-detail":
      self.action_close_detail()
    elif button_id == "end":
      self.query_one("#end-confirm", Vertical).display = True
    elif button_id == "cancel-end":
      self.query_one("#end-confirm", Vertical).display = False
    elif button_id == "confirm-end":
      self.query_one("#end-confirm", Vertical).display = False
      self.query_one("#end", Button).label = "ENDING"
    elif button_id == "send":
      self._append_prompt()

  def on_input_submitted(self, event: Input.Submitted) -> None:
    """Echo submitted steering text into the visible trace."""
    self._append_prompt()

  def action_close_detail(self) -> None:
    """Close the detail split and clear the active anomaly card."""
    confirm = self.query_one("#end-confirm", Vertical)
    if confirm.display:
      confirm.display = False
      return

    self.selected_anomaly = ""
    self._sync_anomaly_detail()

  def select_anomaly(self, anomaly_id: str) -> None:
    """Open a clicked anomaly, or close it if it is already selected."""
    self.selected_anomaly = "" if self.selected_anomaly == anomaly_id else anomaly_id
    self._sync_anomaly_detail()

  def _animate_subagents(self) -> None:
    """Rotate subagent spinners so running cards visibly update."""
    self.spin += 1
    for offset, (agent_id, chip, _, _, _) in enumerate(SUBAGENTS):
      frame = SPINNER_FRAMES[(self.spin + offset) % len(SPINNER_FRAMES)]
      for status in self.query(f"#status-{agent_id}"):
        status.update(_subagent_status(frame, agent_id, chip))

  def _sync_anomaly_detail(self) -> None:
    """Refresh card state and selected detail pane content."""
    selected = self.anomalies.get(self.selected_anomaly)
    for card in self.query(AnomalyCard):
      card.render_card(active=card.anomaly["aid"] == self.selected_anomaly)

    detail = self.query_one("#detail-pane", Vertical)
    detail.display = selected is not None
    if selected is None:
      return

    self.query_one("#detail-header-text", Static).update(f"ANOMALY DETAIL  [#504945]{selected['aid']}[/]")
    self.query_one("#detail-title", Static).update(f"[#fb4934]●[/] [#ebdbb2]{selected['title']}[/]")
    self.query_one("#detail-meta", Static).update(
      f"subagent [#ebdbb2]{selected['sub']}[/] | chip [#8ec07c]{selected['chip']}[/]\n"
      f"commit [#d3869b]{selected['commit']}[/] | metric [{selected['metric_color']}]{selected['metric']}[/] | at [#bdae93]{selected['ts']}[/]"
    )
    self.query_one("#detail-body", Static).update(selected["detail"])

  def _append_prompt(self) -> None:
    """Append one local user steer and one agent acknowledgement to the trace."""
    prompt = self.query_one("#prompt", Input)
    text = prompt.value.strip()
    if not text:
      return

    trace = self.query_one("#trace", ScrollableContainer)
    trace.mount(_trace_line("❯", "#fabd2f", "steer", "#b8bb26", text, "#ebdbb2"))
    trace.mount(_trace_line("◈", "#d3869b", "thinking", "#b16286", "folding that steer into the current benchmark plan", "#bdae93"))
    prompt.value = ""

  def _subagents(self) -> Vertical:
    """Render the left subagent status column."""
    cards = [_subagent_card(*subagent) for subagent in SUBAGENTS]
    return Vertical(
      Static("SUBAGENTS", classes="section-title"),
      ScrollableContainer(*cards, classes="scroll-pane"),
      id="subagents",
    )

  def _main(self) -> Vertical:
    """Render the trace/detail split and bottom steer row."""
    return Vertical(
      Horizontal(self._trace_pane(), self._detail_pane(), id="main-split"),
      Horizontal(
        Static("❯", id="prompt-icon"),
        Input(placeholder="Ask the agent", id="prompt"),
        Button("SEND ↵", id="send"),
        Button("END", id="end"),
        id="prompt-row",
      ),
      id="main",
    )

  def _end_overlay(self) -> Vertical:
    """Render the hidden End confirmation overlay."""
    return Vertical(
      Static("END RUN?", id="end-confirm-title"),
      Static("Stop after the current benchmark finishes?", id="end-confirm-body"),
      Horizontal(
        Button("CANCEL", id="cancel-end"),
        Button("END", id="confirm-end"),
        id="end-confirm-actions",
      ),
      id="end-confirm",
    )

  def _trace_pane(self) -> Vertical:
    """Render the central trace pane as an agent conversation."""
    trace_lines = [_trace_line(*line) for line in TRACE_ROWS]
    return Vertical(
      Static("TRACE", classes="section-title"),
      ScrollableContainer(*trace_lines, id="trace", classes="scroll-pane"),
      id="trace-pane",
    )

  def _detail_pane(self) -> Vertical:
    """Render the split anomaly detail pane populated by selection state."""
    return Vertical(
      Horizontal(
        Static("ANOMALY DETAIL", id="detail-header-text"),
        Button("× close", id="close-detail"),
        id="detail-header",
      ),
      Static("", id="detail-title"),
      Static("", id="detail-meta"),
      Static("", id="detail-body"),
      Static("linked 3 experiments · dedup hash 0x9f2a4c · stored in anomalies table · click card again or × to close", id="detail-footer"),
      id="detail-pane",
    )

  def _anomalies(self) -> Vertical:
    """Render the right anomaly list column."""
    cards = [AnomalyCard(anomaly) for anomaly in ANOMALIES]
    return Vertical(
      Static("ANOMALIES                    6", classes="section-title alert"),
      ScrollableContainer(*cards, classes="scroll-pane"),
      id="anomalies",
    )


def _subagent_card(agent_id: str, chip: str, done: int, total: int, elapsed: str) -> Vertical:
  """Render one compact subagent card with a fixed-width progress bar."""
  return Vertical(
    Static(_subagent_status(SPINNER_FRAMES[0], agent_id, chip), id=f"status-{agent_id}"),
    Static(f"[#8ec07c]{_progress_bar(done, total)}[/] [#928374]{done}/{total}[/]"),
    Static(f"[#504945]⏱ {elapsed}[/]"),
    classes="subagent-card",
  )


def _subagent_status(frame: str, agent_id: str, chip: str) -> str:
  """Render a colored subagent status line with the current spinner frame."""
  return f"[#8ec07c]{frame}[/] [#ebdbb2]{agent_id}[/]  [#8ec07c]{chip}[/]"


def _progress_bar(done: int, total: int) -> str:
  """Convert done/total counts into a fixed-width bar."""
  width = 12
  filled = round(width * done / total)
  return "█" * filled + "░" * (width - filled)


def _trace_line(glyph: str, glyph_color: str, tag: str, tag_color: str, text: str, text_color: str) -> Static:
  """Render one HTML-style trace row with a glyph, action tag, and body."""
  classes = "trace-line anomaly" if tag == "ANOMALY" else "trace-line"
  return Static(f"[{glyph_color}]{glyph}[/] [{tag_color}]{tag}[/] [{text_color}]{escape(text)}[/]", classes=classes)


if __name__ == "__main__":
  PerferoxTUI().run()
