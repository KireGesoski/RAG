import re

class GuardBlocked(Exception):
    """Raised when guardrails block unsafe input or output."""
    pass

class SimpleGuard:
    def __init__(self):
        # PII patterns
        self.pii_patterns = [
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
            r"\b(?:\d[ -]*?){13,16}\b"                          # credit card
        ]
        # Dangerous code patterns
        self.malware_patterns = [
            r"rm -rf", r"eval\(", r"exec\(", r"import os", r"base64\.b64decode"
        ]

    def mask_pii(self, text):
        masked = text
        for p in self.pii_patterns:
            masked = re.sub(p, "[SENSITIVE_DATA_DELETED]", masked)
        return masked

    def detect_malware(self, text):
        hits = []
        for p in self.malware_patterns:
            if re.search(p, text, re.IGNORECASE):
                hits.append(p)
        return hits

    def run(
        self,
        user_input,
        llm_callable,
        mask_input_pii=True,
        block_on_malware_input=True,
        block_on_malware_output=True,
        block_on_output_pii=True
    ):
        # --- MASK INPUT ---
        safe_input = self.mask_pii(user_input) if mask_input_pii else user_input

        # --- CHECK INPUT ---
        if block_on_malware_input:
            hits = self.detect_malware(user_input)
            if hits:
                raise GuardBlocked(f"Blocked: malware detected in input: {hits}")

        # --- CALL LLM ---
        output_raw = llm_callable(safe_input)

        # --- ENSURE STRING ---
        if not isinstance(output_raw, str):
            try:
                output = str(getattr(output_raw, "text", output_raw))
            except Exception:
                raise GuardBlocked("LLM output is not convertible to string")
        else:
            output = output_raw

        # --- CHECK OUTPUT PII ---
        if block_on_output_pii:
            for p in self.pii_patterns:
                if re.search(p, output):
                    raise GuardBlocked("Blocked: PII detected in output")

        # --- CHECK OUTPUT MALWARE ---
        if block_on_malware_output:
            hits = self.detect_malware(output)
            if hits:
                raise GuardBlocked(f"Blocked: malware detected in output: {hits}")

        return output
