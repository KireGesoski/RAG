from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import os
import sys
import textwrap
from guardrails import Guard, OnFailAction
from guardrails.hub import RegexMatch

guard = Guard().use(
    RegexMatch, regex=r"\(?\d{3}\)?-? *\d{3}-? *-?\d{4}", on_fail=OnFailAction.EXCEPTION
)

#guard.validate("123-456-7890")  # Guardrail passes

try:
    guard.validate("123-789-0000")  # Guardrail fails
except Exception as e:
    print('Kire', e)
os._exit(1)

if __name__ == "__main__":
    main()