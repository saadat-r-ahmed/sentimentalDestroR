from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from destror.utils.logging import get_git_sha


@dataclass
class AttackResult:
    """One adversarial example record — matches the JSON schema in revision.md §1.3."""

    id: str
    dataset: str
    model: str
    attack: str

    original: str
    adversarial: str

    orig_label: int | str
    orig_pred: int | str
    adv_pred: int | str

    orig_conf: float
    adv_conf: float

    n_queries: int
    perturbation_pct: float

    labse_sim: float = -1.0
    perplexity_delta: float = 0.0

    success: bool = False
    seed: int = 42
    git_sha: Optional[str] = field(default_factory=get_git_sha)
    elapsed_sec: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Type alias for the victim model callable
VictimFn = Callable[[str], tuple[int | str, float]]


class Attack(ABC):
    """Abstract base class for all destroR and baseline attacks.

    Subclasses must implement `attack_one`. The `attack_dataset` loop
    handles skipping mispredictions, tracking queries, and W&B logging.
    """

    name: str = "base"

    def __init__(self, victim: VictimFn, seed: int = 42):
        self.victim = victim
        self.seed = seed

    @abstractmethod
    def attack_one(
        self,
        text: str,
        orig_pred: int | str,
        orig_conf: float,
    ) -> tuple[str, int | str, float, int]:
        """Return (adversarial_text, adv_pred, adv_conf, n_queries)."""

    def attack_dataset(
        self,
        records: list[dict],
        dataset_name: str,
        model_name: str,
        max_samples: Optional[int] = None,
    ) -> list[AttackResult]:
        """Run the attack over a list of {id, text, label} dicts."""
        from tqdm import tqdm
        from destror.metrics.core import perturbation_rate

        results: list[AttackResult] = []
        subset = records if max_samples is None else records[:max_samples]

        for rec in tqdm(subset, desc=f"{self.name} @ {model_name}/{dataset_name}"):
            t0 = time.perf_counter()
            orig_pred, orig_conf = self.victim(rec["text"])

            # Skip examples the model already gets wrong
            if orig_pred != rec["label"]:
                results.append(
                    AttackResult(
                        id=str(rec["id"]),
                        dataset=dataset_name,
                        model=model_name,
                        attack=self.name,
                        original=rec["text"],
                        adversarial=rec["text"],
                        orig_label=rec["label"],
                        orig_pred=orig_pred,
                        adv_pred=orig_pred,
                        orig_conf=orig_conf,
                        adv_conf=orig_conf,
                        n_queries=1,
                        perturbation_pct=0.0,
                        success=False,
                        seed=self.seed,
                        meta={"skipped": "misprediction"},
                    )
                )
                continue

            adv_text, adv_pred, adv_conf, n_queries = self.attack_one(
                rec["text"], orig_pred, orig_conf
            )
            elapsed = time.perf_counter() - t0

            results.append(
                AttackResult(
                    id=str(rec["id"]),
                    dataset=dataset_name,
                    model=model_name,
                    attack=self.name,
                    original=rec["text"],
                    adversarial=adv_text,
                    orig_label=rec["label"],
                    orig_pred=orig_pred,
                    adv_pred=adv_pred,
                    orig_conf=orig_conf,
                    adv_conf=adv_conf,
                    n_queries=n_queries,
                    perturbation_pct=perturbation_rate(rec["text"], adv_text),
                    success=adv_pred != orig_pred,
                    seed=self.seed,
                    elapsed_sec=elapsed,
                )
            )

        return results
