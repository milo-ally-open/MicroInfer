"""推論層錯誤，不依賴 HTTP 框架。"""


class InferenceError(Exception):
    """可映射至 API 錯誤回應的推論錯誤。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
