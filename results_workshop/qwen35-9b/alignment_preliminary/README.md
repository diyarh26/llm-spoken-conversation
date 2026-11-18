# Preliminary alignment run (NOT comparable with the paper yet)

ALIGN was run on the Qwen3.5-9B C1/C2 conversations together with the 50 Switchboard
calls only. ALIGN filters the vocabulary (frequency cut-offs) over the whole set of
conversations processed together, so the Switchboard reference here (0.453) differs from
the validated run (0.550), which processed Switchboard together with the Vicuna data.

Use only for relative comparisons within this run. The final alignment numbers will come
from one combined run (Vicuna + Qwen + Switchboard) after the environment is verified to
reproduce the validated Switchboard value.
