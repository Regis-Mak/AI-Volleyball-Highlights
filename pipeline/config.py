"""config.yaml -> typed settings. Unknown keys are rejected so typos surface early."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _names(v: object) -> object:
    """Player names. Older rosters gave each one as {name, description} with
    their shirt; only the name is kept, since shirts change every week."""
    if isinstance(v, list):
        return [p.get("name", "") if isinstance(p, dict) else p for p in v]
    return v


Roster = Annotated[list[str], BeforeValidator(_names)]


class IngestCfg(Strict):
    proxy_height: int = 720
    proxy_fps: int = 30
    audio_filter: str = "highpass=f=120,afftdn=nf=-25"


class RenderCfg(Strict):
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"
    height: int = 2160
    fps: int = 30
    fade_s: float = 0.25
    # Between clips: dip to black, cross-dissolve, or a swipe (the next clip
    # slides in). Shares of the cuts that get each; clips with nametags dip.
    transitions: dict[str, float] = {"fade": 0.5, "dissolve": 0.35, "swipe": 0.15}
    transition_s: float = 0.5
    audio_bitrate: str = "192k"
    default_pre_s: float = 6
    default_post_s: float = 6
    nvenc_args: list[str] = ["-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    x264_args: list[str] = ["-preset", "veryfast", "-crf", "20"]


class WhisperCfg(Strict):
    enabled: bool = True
    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "float16"
    language: str | None = "en"
    vad_filter: bool = True


class PannsCfg(Strict):
    enabled: bool = True
    device: Literal["cuda", "cpu"] = "cuda"
    window_s: float = 2.0
    batch_size: int = 32
    min_full_scale: float = 0.3
    classes: dict[str, list[str]] = Field(default_factory=dict)


class LoudnessCfg(Strict):
    baseline_s: int = 60
    full_scale_db: float = 12


class PeaksCfg(Strict):
    min_separation_s: float = 20
    top_n: int = 40


class AudioCfg(Strict):
    whisper: WhisperCfg = WhisperCfg()
    panns: PannsCfg = PannsCfg()
    loudness: LoudnessCfg = LoudnessCfg()
    keywords: list[str] = []
    weights: dict[str, float] = {}
    smooth_s: int = 3
    peaks: PeaksCfg = PeaksCfg()


class MotionCfg(Strict):
    sample_fps: int = 4
    idle_fraction: float = 0.35
    keep_if_reaction: float = 0.12


class VlmCfg(Strict):
    model: str = "qwen3-vl:8b-instruct"
    host: str = "http://localhost:11434"
    window_s: float = 24
    hop_s: float = 16
    base_fps: float = 1
    burst_fps: float = 4
    burst_s: float = 3
    burst_min_reaction: float = 0.15
    max_frames: int = 40
    frame_width: int = 640
    num_ctx: int = 16384
    temperature: float = 0.1
    max_windows: int = 0
    timeout_s: float = 300
    max_answer_tokens: int = 1024     # an answer is ~200; stops one that runs on and on


class ReviewCfg(Strict):
    top_n: int = 20
    pre_s: float = 8
    post_s: float = 4
    label_clips: bool = True
    font_file: str = "auto"


class SessionCfg(Strict):
    join_tolerance_s: float = 2.0


class ClipShape(Strict):
    pre_s: float
    post_s: float


def _default_shapes() -> dict[str, ClipShape]:
    return {
        "great_play": ClipShape(pre_s=8, post_s=4),
        "blooper": ClipShape(pre_s=6, post_s=6),
        "celebration": ClipShape(pre_s=4, post_s=6),
        "chaos": ClipShape(pre_s=8, post_s=4),
        "rally": ClipShape(pre_s=10, post_s=3),
        "conversation": ClipShape(pre_s=2, post_s=16),
        "reaction": ClipShape(pre_s=8, post_s=4),
    }


class FuseCfg(Strict):
    # rally = how long the rally lasts (rally_short_s -> 0, rally_long_s -> 1),
    # attack = the hardest spike Qwen saw / 10.
    weights: dict[str, float] = {"rally": 0.3, "attack": 0.3, "excitement": 0.2, "funny": 0.1,
                                 "reaction": 0.1}
    fun_mix: float = 0.6          # fun = fun_mix x funny/10 + (1 - fun_mix) x laughter
    label_bonus: dict[str, float] = {"great_play": 0.15, "blooper": 0.05, "chaos": 0.05,
                                     "celebration": 0.05, "conversation": 0.0, "rally": 0.0,
                                     "reaction": 0.0}
    audio_only_factor: float = 0.8
    audio_min_reaction: float = 0.3
    reaction_lookback_s: float = 2
    reaction_lookahead_s: float = 8
    merge_within_s: float = 10
    clip: dict[str, ClipShape] = Field(default_factory=_default_shapes)
    extend_while_reaction: float = 0.25
    max_extend_s: float = 6
    min_clip_s: float = 6
    max_clip_s: float = 20
    rally_short_s: float = 5      # rallies this short get nothing for length...
    rally_long_s: float = 20      # ...this long or longer get the full rally weight
    max_rally_clip_s: float = 60  # a whole rally (with its celebration) up to this long
    # These labels get the whole rally: from just before the serve, through the
    # celebration, stopping just before the next serve.
    whole_rally_labels: list[str] = ["great_play", "rally", "chaos"]
    serve_pre_s: float = 1.5      # start this long before the serve (the toss)
    celebrate_min_s: float = 4    # keep at least this long after the ball dies...
    celebrate_max_s: float = 15   # ...and up to this long while the group still reacts
    next_serve_gap_s: float = 0.5 # stop this long before the next serve


class ReelCfg(Strict):
    target_min: float = 20
    min_min: float = 18
    max_min: float = 22
    max_fraction: float = 0.25
    min_score: float = 0.25
    merge_gap_s: float = 1.5
    max_merged_s: float = 70
    order: Literal["chronological", "best_first"] = "chronological"


class ThemeCfg(Strict):
    title: str
    labels: list[str] = []
    rank_by: Literal["score", "hype", "fun"] = "score"
    min_value: float = 0.4


def _default_themes() -> dict[str, ThemeCfg]:
    return {
        "top_plays": ThemeCfg(title="top plays", labels=["great_play", "chaos"],
                              rank_by="hype", min_value=0.5),
        "bloopers": ThemeCfg(title="bloopers", labels=["blooper"], rank_by="fun",
                             min_value=0.4),
        "funniest": ThemeCfg(title="funniest moments", rank_by="fun", min_value=0.5),
        "celebrations": ThemeCfg(title="celebrations", labels=["celebration"],
                                 rank_by="score", min_value=0.35),
        "conversations": ThemeCfg(title="conversations", labels=["conversation"],
                                  rank_by="score", min_value=0.1),
    }


class ClipThatCfg(Strict):
    """When someone says "clip that", that play also becomes its own video."""
    enabled: bool = True
    # Regular expressions, matched against what Whisper heard (any case).
    phrases: list[str] = [r"\bclip[\s,.!-]*(?:that|it|this)\b", r"\bthat'?s a clip\b",
                          r"\bclip-?worthy\b"]
    lookback_s: float = 25        # the play it means ended at most this long before
    before_s: float = 15          # no tagged play found: start this long before the words
    after_s: float = 4            # keep this long after the words
    max_s: float = 45
    min_gap_s: float = 20         # "clip that! clip that!" is one clip
    reel_bonus: float = 0.3       # the play also counts this much more for the reel


class CommentsCfg(Strict):
    """Unhinged comments: Qwen reads the transcript and rates what people said.
    A reaction right after a line ("whaaat?!", laughter) means it counts for more."""
    enabled: bool = True
    model: str = ""               # "" = vlm.model (no extra download)
    lines_per_read: int = 30      # transcript lines Qwen reads at a time...
    overlap_lines: int = 6        # ...overlapping, so a line and its reaction are read together
    max_answer_tokens: int = 1500
    # A reaction like these within cue_window_s after a comment makes it count
    # cue_boost more (regular expressions, any case, on what Whisper heard).
    cues: list[str] = [r"\bwh+a{2,}t+\b", r"(?:^|[.!?]\s+)wh+a+t+\s*[?!]+",
                       r"\bwhat the heck\b", r"\b(?:bro|yo|dude|bruh),? what\b",
                       r"\bexcuse me\b", r"\byou can'?t (?:just )?say that\b",
                       r"\bwhy would you (?:say|do) that\b", r"\bwho says that\b",
                       r"\bno shot\b", r"\bbr+u+h+\b", r"\bare you serious\b",
                       r"\bthat'?s (?:crazy|insane|wild)\b", r"\byou'?re (?:crazy|insane|wild)\b"]
    cue_window_s: float = 8
    cue_boost: float = 2
    laugh_boost: float = 1        # clear laughter right after it
    min_unhinged: float = 5       # 0-10: this or more, after boosts, goes in the reel
    max_reel_share: float = 0.3   # comments sure of a place take at most this share of the reel
    score_weight: float = 0.4     # x unhinged/10, added to the moment's score
    pre_s: float = 1.5            # clip starts this long before the comment...
    post_s: float = 3             # ...and runs this long past the reaction
    max_clip_s: float = 30


class ShortsCfg(Strict):
    enabled: bool = False
    min_s: float = 30
    max_s: float = 60
    max_per_theme: int = 2
    min_moments: int = 2
    order: Literal["chronological", "best_first"] = "best_first"
    themes: dict[str, ThemeCfg] = Field(default_factory=_default_themes)


class TrustCfg(Strict):
    hand: float = 1.0          # your grade on the Grade screen
    cut: float = 0.6           # keep = n in picks.csv
    trim: float = 0.5          # changed clip start/end in picks.csv
    kept: float = 0.3          # left in when you edited picks.csv
    agree: float = 0.4         # audio and vision clearly agree (tool-made)


class RankerCfg(Strict):
    enabled: bool = True
    min_labels: int = 30
    blend_full_at: int = 300
    model: Literal["auto", "logistic", "lightgbm"] = "auto"
    lightgbm_from: int = 400


class TestSetCfg(Strict):
    target: int = 100
    min_to_check: int = 20


class GradeCfg(Strict):
    per_session: int = 10
    port: int = 8765


class ExamplesCfg(Strict):
    enabled: bool = True
    model: str = "google/siglip-base-patch16-224"   # picture fingerprints (downloaded once)
    device: str = "auto"          # auto (GPU when there is one) | cuda | cpu
    count: int = 4                # past moments shown to Qwen per window
    min_examples: int = 20        # hints start once this many labelled moments exist
    min_similarity: float = 0.2   # how alike (0-1) a past moment must look to be shown
    min_trust: float = 0.5        # only labels at least this trusted (hand, cut, trim)


class TuneCfg(Strict):
    enabled: bool = True
    min_labels: int = 40          # start tuning the scoring once this many moments have labels
    min_trims: int = 5            # a label's clip length is tuned after this many trims


class FinetuneCfg(Strict):
    min_labels: int = 500         # graded moments before fine-tuning Qwen is worth a try
    min_trust: float = 0.5        # labels used as answers: your grades, cuts and trims
    val_share: float = 0.1        # share of training examples held back to watch training


class LearningCfg(Strict):
    enabled: bool = True
    library_dir: Path = Path("library")
    save_frames: bool = True
    trust: TrustCfg = TrustCfg()
    self_made_cap: float = 0.5
    ranker: RankerCfg = RankerCfg()
    test_set: TestSetCfg = TestSetCfg()
    grade: GradeCfg = GradeCfg()
    examples: ExamplesCfg = ExamplesCfg()
    tune: TuneCfg = TuneCfg()
    finetune: FinetuneCfg = FinetuneCfg()


class TrackCfg(Strict):
    model: str = "yolo11s.pt"     # Ultralytics person detector, downloaded once into models/
    models_dir: Path = Path("models")
    device: str = "auto"          # auto (GPU when there is one) | cuda | cpu
    fps: float = 10               # frames per second looked at
    imgsz: int = 1280
    conf: float = 0.25


class NametagsCfg(Strict):
    enabled: bool = True
    seconds: float = 18
    shorts: bool = False
    text_height: float = 0.02     # letter height as a share of the video height
    box_opacity: float = 0.45
    text_color: str = "#ffffff"
    font: str = "builtin"         # builtin pixel font, or a .ttf / .otf / .ttc file
    fade_s: float = 0.4
    min_visible_s: float = 2.0


class VerticalCfg(Strict):
    shorts: Literal["landscape", "vertical", "both"] = "landscape"
    width: int = 1080             # the shape (9:16); also the size when resize is on
    height: int = 1920
    resize: bool = False          # off: keep the crop's own pixels; on: resize to width x height
    dead_zone: float = 0.3        # share of the crop the action can move in before it pans
    max_pan: float = 0.35         # fastest pan, in frame widths per second
    smooth_s: float = 0.6
    zoom: float = 1.0             # 1 = full height; higher = closer (fewer pixels), following players up and down


class Config(Strict):
    work_dir: Path = Path("work")
    output_dir: Path = Path("output")
    players: Roster = []          # names offered on the Who is who screen
    ingest: IngestCfg = IngestCfg()
    render: RenderCfg = RenderCfg()
    audio: AudioCfg = AudioCfg()
    motion: MotionCfg = MotionCfg()
    vlm: VlmCfg = VlmCfg()
    review: ReviewCfg = ReviewCfg()
    session: SessionCfg = SessionCfg()
    fuse: FuseCfg = FuseCfg()
    reel: ReelCfg = ReelCfg()
    shorts: ShortsCfg = ShortsCfg()
    clip_that: ClipThatCfg = ClipThatCfg()
    comments: CommentsCfg = CommentsCfg()
    learning: LearningCfg = LearningCfg()
    tracking: TrackCfg = TrackCfg()
    nametags: NametagsCfg = NametagsCfg()
    vertical: VerticalCfg = VerticalCfg()
    config_path: Path | None = None   # set by load_config


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(data)
    base = path.resolve().parent
    if not cfg.work_dir.is_absolute():
        cfg.work_dir = base / cfg.work_dir
    if not cfg.output_dir.is_absolute():
        cfg.output_dir = base / cfg.output_dir
    if not cfg.learning.library_dir.is_absolute():
        cfg.learning.library_dir = base / cfg.learning.library_dir
    if not cfg.tracking.models_dir.is_absolute():
        cfg.tracking.models_dir = base / cfg.tracking.models_dir
    cfg.config_path = path.resolve()
    return cfg
