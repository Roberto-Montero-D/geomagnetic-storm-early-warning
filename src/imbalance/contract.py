"""Frozen Phase 4 imbalance-screening contract."""
from dataclasses import dataclass
from src.feature_screening.freeze import PHASE3_SELECTED_FEATURES

PHASE4_FEATURES = PHASE3_SELECTED_FEATURES
PHASE4_SCREENING_FOLD = "initial_screening"
PHASE4_CONFIRMATION_FOLDS = ("walk_forward_1", "walk_forward_2")
PHASE4_MAX_FAR_PER_DAY = 0.2
PHASE4_RANDOM_STATE = 42
PHASE4_EXTRATREES_PARAMS = {"n_estimators":100,"max_depth":10,"random_state":42}

@dataclass(frozen=True)
class ImbalanceExperiment:
    name: str
    strategy: str
    parameter: object = None

PHASE4_EXPERIMENTS = (
    ImbalanceExperiment("none","none"),
    *(ImbalanceExperiment(f"class_weight_{w}","class_weighting",w) for w in (1,3,5,10,20,50)),
    *(ImbalanceExperiment(f"undersample_{r}_to_1","random_undersampling",f"{r}:1") for r in (10,5,2)),
    *(ImbalanceExperiment(f"smote_k{k}","smote",k) for k in (3,5,7)),
    *(ImbalanceExperiment(f"borderline_smote_k{k}","borderline_smote",k) for k in (3,5,7)),
    ImbalanceExperiment("smote_enn","smote_enn"),
)
PHASE4_EXPERIMENT_NAMES = tuple(x.name for x in PHASE4_EXPERIMENTS)
PHASE4_ADVANCE_COUNT = 3
PHASE4_RANKING_KEYS = ("operationally_feasible","event_recall","pr_auc","false_alarm_rate_per_day","experiment_order")
PHASE4_RANKING_ASCENDING = (False,False,False,True,True)
PHASE4_CONFIRMATION_RANKING_KEYS = ("confirmation_feasible","minimum_event_recall","mean_event_recall","mean_pr_auc","mean_false_alarm_rate_per_day","experiment_order")
PHASE4_CONFIRMATION_RANKING_ASCENDING = (False,False,False,False,True,True)
PHASE4_CONTRACT_STATUS = "frozen"
