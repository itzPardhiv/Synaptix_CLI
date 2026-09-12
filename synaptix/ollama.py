"""Safe Ollama client for SYNAPTIX.

SYNAPTIX only connects to the existing Ollama service.
It never starts Ollama itself.
"""

from typing import Any, Dict, Optional

import requests

from .config import SETTINGS


class OllamaClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (
            base_url or SETTINGS.ollama_base_url
        ).rstrip("/")

        self.timeout = (
            timeout or SETTINGS.ollama_timeout
        )

        self.session = requests.Session()

    def health(self) -> bool:
        """Check whether Ollama is already running."""

        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=(3, 8),
            )

            return response.ok

        except requests.RequestException:
            return False

    def models(self):
        """Return installed Ollama models."""

        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=(3, 10),
            )

            response.raise_for_status()

            data = response.json()

            return [
                model.get("name", "")
                for model in data.get("models", [])
                if model.get("name")
            ]

        except (
            requests.RequestException,
            ValueError,
        ):
            return []

    def chat(
        self,
        messages,
        model=None,
        options=None,
    ) -> str:

        payload: Dict[str, Any] = {
            "model": model or SETTINGS.model,
            "messages": messages,
            "stream": False,

            "options": options or {
                "temperature": 0.15,
                "top_p": 0.90,
                "repeat_penalty": 1.08,
                "num_ctx": SETTINGS.context_window,
            },
        }

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=(5, self.timeout),
        )

        response.raise_for_status()

        data = response.json()

        return (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )